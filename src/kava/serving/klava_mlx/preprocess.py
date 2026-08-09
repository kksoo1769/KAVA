"""SigLIP-2 NaFlex 이미지 전처리를 NumPy와 Pillow로 구현."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

__all__ = [
    "PATCH_SIZE",
    "DEFAULT_MAX_NUM_PATCHES",
    "IMAGE_MEAN",
    "IMAGE_STD",
    "RESCALE_FACTOR",
    "get_image_size_for_max_num_patches",
    "aa_bilinear_weights",
    "aa_bilinear_resize",
    "convert_image_to_patches",
    "pad_along_first_dim",
    "preprocess_image",
    "PreprocessResult",
]

# 전처리 기본값
PATCH_SIZE = 16
DEFAULT_MAX_NUM_PATCHES = 784 # runs/.../meta.json의 siglip_num_patches
IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.5, 0.5, 0.5)
RESCALE_FACTOR = 0.00392156862745098 # preprocessor_config.json (= 1/255)
MAX_RESIZE_TEMP_BYTES = 64 << 20

# D3. 목표 크기
def get_image_size_for_max_num_patches(
    image_height: int,
    image_width: int,
    patch_size: int = PATCH_SIZE,
    max_num_patches: int = DEFAULT_MAX_NUM_PATCHES,
    eps: float = 1e-5,
) -> tuple[int, int]:
    """최대 패치 수에 맞는 이미지 크기를 계산."""

    def get_scaled_image_size(scale: float, size: int, patch: int) -> int:
        scaled_size = size * scale
        scaled_size = math.ceil(scaled_size / patch) * patch # patch 배수로 올림
        scaled_size = max(patch, scaled_size) # 최소 1패치
        return int(scaled_size)

    scale_min, scale_max = eps / 10, 100.0
    while (scale_max - scale_min) >= eps:
        scale = (scale_min + scale_max) / 2
        target_height = get_scaled_image_size(scale, image_height, patch_size)
        target_width = get_scaled_image_size(scale, image_width, patch_size)
        num_patches = (target_height / patch_size) * (target_width / patch_size)
        if num_patches <= max_num_patches:
            scale_min = scale
        else:
            scale_max = scale

    scale = scale_min
    target_height = get_scaled_image_size(scale, image_height, patch_size)
    target_width = get_scaled_image_size(scale, image_width, patch_size)
    return target_height, target_width


# D4와 D11에서 같은 리샘플러를 사용한다
def aa_bilinear_weights(in_size: int, out_size: int) -> tuple[np.ndarray, np.ndarray]:
    """1차원 antialias bilinear 계수를 계산."""
    if in_size <= 0 or out_size <= 0:
        raise ValueError(f"크기는 양수여야 한다: in={in_size} out={out_size}")

    scale = in_size / out_size
    filter_scale = scale if scale >= 1.0 else 1.0
    support = filter_scale # bilinear 필터의 support 는 1.0 이므로 support = filter_scale
    invscale = 1.0 / filter_scale

    # C 구현의 interp_size 와 같은 상한. 실제 xsize 는 경계에서 더 작을 수 있다.
    k = int(math.ceil(support)) * 2 + 1
    k = min(k, in_size)

    starts = np.zeros(out_size, dtype=np.int64)
    weights = np.zeros((out_size, k), dtype=np.float64)

    for i in range(out_size):
        center = scale * (i + 0.5)
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), in_size) - xmin
        if xmax <= 0: # 잘못된 계수로 NaN이 생기지 않도록 확인한다
            raise RuntimeError(
                f"리샘플 계수가 비었다: in={in_size} out={out_size} i={i} "
                f"center={center} xmin={xmin}"
            )
        if xmax > k:
            raise RuntimeError(f"xsize({xmax}) > k({k}): 계수 버퍼 상한 계산이 틀렸다")
        total = 0.0
        for j in range(xmax):
            w = 1.0 - abs((j + xmin - center + 0.5) * invscale)
            w = w if w > 0.0 else 0.0
            weights[i, j] = w
            total += w
        if total == 0.0:
            raise RuntimeError(f"리샘플 계수 합이 0 이다: in={in_size} out={out_size} i={i}")
        weights[i, :xmax] /= total
        starts[i] = xmin

    return starts, weights


def _resize_axis(arr: np.ndarray, axis: int, out_size: int) -> np.ndarray:
    """arr 의 axis 를 out_size 로 antialias bilinear 리샘플(float64 누적)."""
    in_size = arr.shape[axis]
    if in_size == out_size:
        return arr
    starts, weights = aa_bilinear_weights(in_size, out_size)
    k = weights.shape[1]

    # gather 인덱스: (out_size, k). 경계 밖은 clip 하되 그 자리 weight 는 0 이라 기여 없음.
    idx = starts[:, None] + np.arange(k)[None, :]
    idx = np.clip(idx, 0, in_size - 1)

    moved = np.moveaxis(arr, axis, -1)  # (..., in_size)
    rows = int(np.prod(moved.shape[:-1]))
    step = max(1, MAX_RESIZE_TEMP_BYTES // max(1, rows * k * 8))

    out = np.empty(moved.shape[:-1] + (out_size,), dtype=np.float64)
    for a in range(0, out_size, step):
        b = min(a + step, out_size)
        gathered = moved[..., idx[a:b]]  # (..., b-a, k)
        out[..., a:b] = np.einsum("...ok,ok->...o", gathered, weights[a:b])
    return np.moveaxis(out, -1, axis)


def aa_bilinear_resize(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """float 배열을 antialias bilinear 방식으로 리샘플."""
    x = np.asarray(arr, dtype=np.float64)
    x = _resize_axis(x, -2, out_h)
    x = _resize_axis(x, -1, out_w)
    return x


def _resize_uint8_pil(image_chw_u8: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Pillow로 uint8 CHW 이미지를 리사이즈."""
    from PIL import Image # 지연 import (numpy 만으로 쓰는 경로를 막지 않기 위해)

    hwc = np.ascontiguousarray(np.transpose(image_chw_u8, (1, 2, 0)))
    if hwc.shape[2] != 3:
        raise ValueError(f"RGB 3채널이어야 한다: shape={hwc.shape}")
    pil = Image.fromarray(hwc, mode="RGB")
    # reducing_gap을 지정하지 않아 Pillow의 사전 축소를 막는다
    resized = pil.resize((out_w, out_h), resample=Image.BILINEAR, reducing_gap=None)
    out = np.asarray(resized, dtype=np.uint8)
    return np.ascontiguousarray(np.transpose(out, (2, 0, 1)))


def _resize_uint8_numpy_aa(image_chw_u8: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """NumPy antialias 구현으로 uint8 CHW 이미지를 리사이즈."""
    x = _resize_axis(image_chw_u8, -2, out_h)
    x = _resize_axis(x, -1, out_w)
    x = np.rint(x)
    x = np.clip(x, 0, 255)
    return x.astype(np.uint8)


# D7과 D8 패치 추출 및 패딩
def convert_image_to_patches(image_chw: np.ndarray, patch_size: int) -> np.ndarray:
    """CHW 이미지를 패치 배열로 변환."""
    num_channels, image_height, image_width = image_chw.shape
    nh = image_height // patch_size
    nw = image_width // patch_size
    patched = image_chw.reshape(num_channels, nh, patch_size, nw, patch_size)
    patched = patched.transpose(1, 3, 2, 4, 0) # (nh, nw, ph, pw, C)
    return patched.reshape(nh * nw, -1)


def pad_along_first_dim(
    array: np.ndarray, target_length: int, pad_value: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """SigLIP2 입력에 맞게 패치와 마스크를 구성."""
    current_length = array.shape[0]
    padding_length = target_length - current_length
    if padding_length < 0:
        raise ValueError(
            f"패치 수({current_length})가 max_num_patches({target_length})를 넘었다: "
            "get_image_size_for_max_num_patches 가 보장해야 하는 불변식이 깨졌다."
        )
    mask = np.ones((target_length,), dtype=np.int32)
    if padding_length > 0:
        paddings = [(0, padding_length)] + [(0, 0)] * (array.ndim - 1)
        array = np.pad(array, paddings, mode="constant", constant_values=pad_value)
        mask[-padding_length:] = 0
    return array, mask


# 최종 진입점
class PreprocessResult(dict):
    """전처리 결과를 담고 속성 접근을 지원하는 dict."""

    __getattr__ = dict.__getitem__ # type: ignore[assignment]


def preprocess_image(
    image: Any,
    patch_size: int = PATCH_SIZE,
    max_num_patches: int = DEFAULT_MAX_NUM_PATCHES,
    image_mean: Iterable[float] = IMAGE_MEAN,
    image_std: Iterable[float] = IMAGE_STD,
    rescale_factor: float = RESCALE_FACTOR,
    resize_backend: str = "numpy_aa",
) -> PreprocessResult:
    """이미지 한 장을 NaFlex 입력 형식으로 전처리."""
    from PIL import Image as PILImage # 지연 import

    # RGB 이미지 로드
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            raise TypeError(f"numpy 입력은 uint8 HWC 여야 한다: dtype={image.dtype}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"numpy 입력은 (H,W,3) 이어야 한다: shape={image.shape}")
        arr_hwc = image
        src_size = (image.shape[1], image.shape[0])
    else:
        if isinstance(image, PILImage.Image):
            pil = image.convert("RGB")
            src_size = pil.size
            arr_hwc = np.asarray(pil, dtype=np.uint8)
        else:
            with PILImage.open(str(image)) as im:
                pil = im.convert("RGB")
                src_size = pil.size
                arr_hwc = np.asarray(pil, dtype=np.uint8)

    # uint8 CHW 변환
    chw = np.ascontiguousarray(np.transpose(arr_hwc, (2, 0, 1)))
    src_h, src_w = chw.shape[1], chw.shape[2]

    # 목표 크기 계산
    tgt_h, tgt_w = get_image_size_for_max_num_patches(
        image_height=src_h, image_width=src_w,
        patch_size=patch_size, max_num_patches=max_num_patches,
    )

    # uint8 이미지 리사이즈
    if (tgt_h, tgt_w) == (src_h, src_w):
        resized = chw
    elif resize_backend == "pil":
        resized = _resize_uint8_pil(chw, tgt_h, tgt_w)
    elif resize_backend == "numpy_aa":
        resized = _resize_uint8_numpy_aa(chw, tgt_h, tgt_w)
    else:
        raise ValueError(f"알 수 없는 resize_backend: {resize_backend!r}")

    # rescale과 normalize 상수를 합친다
    inv = 1.0 / rescale_factor
    mean = np.asarray(list(image_mean), dtype=np.float32) * np.float32(inv)
    std = np.asarray(list(image_std), dtype=np.float32) * np.float32(inv)
    x = resized.astype(np.float32) # :346 .to(dtype=torch.float32)
    x = (x - mean[:, None, None]) / std[:, None, None]

    # 패치 변환
    patches = convert_image_to_patches(x, patch_size)

    # 패딩과 마스크 생성
    patches, mask = pad_along_first_dim(patches, max_num_patches, pad_value=0.0)

    # 공간 크기 생성
    nh = tgt_h // patch_size
    nw = tgt_w // patch_size

    return PreprocessResult(
        pixel_values=np.ascontiguousarray(patches, dtype=np.float32),
        pixel_attention_mask=mask,
        spatial_shapes=np.array([nh, nw], dtype=np.int64),
        meta={
            "source_size_wh": tuple(int(v) for v in src_size),
            "source_hw": (int(src_h), int(src_w)),
            "resized_hw": (int(tgt_h), int(tgt_w)),
            "n_patches": int(nh * nw),
            "max_num_patches": int(max_num_patches),
            "patch_size": int(patch_size),
            "resize_backend": resize_backend,
        },
    )


# 자가 점검 (모델 로드 없음)
def _selftest() -> int:
    """모델을 로드하지 않고 전처리 과정을 점검."""
    from PIL import Image as PILImage

    ok = True

    print("=" * 78)
    print("klava_mlx.preprocess 자가 점검 (모델과 가중치 로드 없음)")
    print("=" * 78)

    print("\n[1] get_image_size_for_max_num_patches (patch=16, max=784)")
    cases = [(666, 1000), (600, 1200), (558, 1540), (953, 1192), (1351, 962), (900, 1200),
             (16, 16), (7, 4000), (1, 1)]
    for h, w in cases:
        th, tw = get_image_size_for_max_num_patches(h, w, 16, 784)
        nh, nw = th // 16, tw // 16
        bad = []
        if th % 16 or tw % 16:
            bad.append("16 배수 아님")
        if nh * nw > 784:
            bad.append(f"패치 수 초과 {nh * nw}")
        if nh < 1 or nw < 1:
            bad.append("패치 0개")
        status = "OK  " if not bad else "FAIL"
        if bad:
            ok = False
        print(f"  {status} {h:5d}x{w:<5d}: {th:4d}x{tw:<5d}  patches {nh:3d}x{nw:<3d}"
              f" = {nh * nw:3d}  {'; '.join(bad)}")

    print("\n[2] aa_bilinear_weights 성질")
    for in_s, out_s in [(16, 28), (16, 8), (100, 33), (33, 100), (7, 7)]:
        starts, w = aa_bilinear_weights(in_s, out_s)
        sums = w.sum(axis=1)
        max_err = float(np.abs(sums - 1.0).max())
        neg = int((w < 0).sum())
        status = "OK  " if (max_err < 1e-12 and neg == 0) else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  {status} in={in_s:4d} out={out_s:4d}  k={w.shape[1]}  "
              f"|sum-1|max={max_err:.2e}  neg={neg}")

    print("\n[3] aa_bilinear_resize vs Pillow float('F' 모드): 같은 알고리즘인가")
    rng = np.random.default_rng(0)
    for (ih, iw), (oh, ow) in [((16, 16), (28, 28)), ((16, 16), (8, 12)),
                               ((16, 16), (17, 46)), ((64, 40), (19, 39))]:
        src = rng.standard_normal((ih, iw)).astype(np.float32)
        mine = aa_bilinear_resize(src, oh, ow)
        ref = np.asarray(
            PILImage.fromarray(src, mode="F").resize((ow, oh), resample=PILImage.BILINEAR),
            dtype=np.float64,
        )
        err = float(np.abs(mine - ref).max())
        status = "OK  " if err < 1e-5 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  {status} {ih}x{iw}: {oh}x{ow}   max|mine-PIL| = {err:.3e}")

    print("\n[4] 패치 순서 (D7): 768 벡터 안에서 채널이 가장 빠르게 변해야 한다")
    # C=3, H=W=32 (=2x2 패치). 값 = c*1_000_000 + y*1000 + x 로 유일하게 식별한다.
    c_idx, y_idx, x_idx = np.meshgrid(
        np.arange(3), np.arange(32), np.arange(32), indexing="ij"
    )
    synth = (c_idx * 1_000_000 + y_idx * 1000 + x_idx).astype(np.float64)
    p = convert_image_to_patches(synth, 16)
    checks = []
    checks.append(("패치 수", p.shape == (4, 768), f"{p.shape}"))
    # 패치 0 (좌상단), 벡터 인덱스 0,1,2 = (ph0,pw0,c0..2)
    checks.append(("v[0:3] = (0,0) 픽셀의 3채널",
                   list(p[0, 0:3]) == [0.0, 1_000_000.0, 2_000_000.0],
                   f"{p[0, 0:3]}"))
    # 인덱스 3 = (ph0, pw1, c0): x=1
    checks.append(("v[3] = (y0,x1) c0", p[0, 3] == 1.0, f"{p[0, 3]}"))
    # 인덱스 16*3 = (ph1, pw0, c0): y=1
    checks.append(("v[48] = (y1,x0) c0", p[0, 48] == 1000.0, f"{p[0, 48]}"))
    # 패치 1 = row-major 이므로 (nh=0, nw=1): x 시작이 16
    checks.append(("패치1 = 오른쪽 이웃 (row-major)", p[1, 0] == 16.0, f"{p[1, 0]}"))
    # 패치 2 = (nh=1, nw=0): y 시작이 16
    checks.append(("패치2 = 아래쪽", p[2, 0] == 16000.0, f"{p[2, 0]}"))
    for name, good, detail in checks:
        if not good:
            ok = False
        print(f"  {'OK  ' if good else 'FAIL'} {name:34s} {detail}")

    print("\n[5] 리사이즈 백엔드 차이 (D5 위험의 크기): 합성 이미지")
    img = rng.integers(0, 256, size=(300, 500, 3), dtype=np.uint8)
    a = preprocess_image(img, resize_backend="pil")
    b = preprocess_image(img, resize_backend="numpy_aa")
    d = np.abs(a["pixel_values"] - b["pixel_values"])
    n_valid = int(a["pixel_attention_mask"].sum())
    print(f"  리사이즈 {a['meta']['source_hw']}: {a['meta']['resized_hw']}  "
          f"유효 패치 {n_valid}/{a['pixel_values'].shape[0]}")
    print(f"  max|pil - numpy_aa| = {d.max():.6f}  (1 LSB = 2/255 = {2 / 255:.6f})")
    print(f"  mean|pil - numpy_aa| = {d.mean():.6f}")
    print(f"  1 LSB 초과 비율      = {(d > 2 / 255 + 1e-6).mean():.6f}")
    print(" : 이 값이 D5 의 '고정소수점 구현 차이' 크기의 대리 지표다. "
          "진짜 값은 verify_vision_parity.py 가 torch 와 직접 비교해 잰다.")

    print("\n[6] 마스크와 모양 계약")
    for shape in [(666, 1000, 3), (558, 1540, 3), (16, 16, 3)]:
        im = rng.integers(0, 256, size=shape, dtype=np.uint8)
        r = preprocess_image(im)
        nh, nw = r["spatial_shapes"]
        good = (
            r["pixel_values"].shape == (784, 768)
            and r["pixel_values"].dtype == np.float32
            and r["pixel_attention_mask"].shape == (784,)
            and int(r["pixel_attention_mask"].sum()) == int(nh) * int(nw)
            and bool((r["pixel_values"][int(nh) * int(nw):] == 0).all())
        )
        if not good:
            ok = False
        print(f"  {'OK  ' if good else 'FAIL'} {shape[0]}x{shape[1]}: "
              f"shapes={tuple(int(v) for v in r['spatial_shapes'])} "
              f"valid={int(r['pixel_attention_mask'].sum())}")

    print("\n" + "=" * 78)
    print("자가 점검 " + ("통과" if ok else "실패"))
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
