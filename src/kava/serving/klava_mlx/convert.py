#!/usr/bin/env python3
"""MLX 모델과 어댑터를 준비하고 검증하는 명령줄 도구."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
from pathlib import Path
from typing import Any, Optional

from kava.paths import REPO_ROOT

from . import default_ckpt_dir, mlx_root

DEFAULT_MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
# KLAVA_CKPT 를 존중한다. 서버(runtime.py)가 읽는 것과 같은 규칙이어야 한다.
DEFAULT_CKPT = Path(default_ckpt_dir())
DEFAULT_ADAPTER = DEFAULT_CKPT / "adapter"

GIB = 1024**3

# 변환 뒤 원본에서 복원할 토크나이저 파일
TOKENIZER_FILES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
)

# 토크나이저와 도구 템플릿 검증에 사용할 고정 입력
_VERIFY_MESSAGES = [
    {"role": "user", "content": "서울 날씨 알려줘"},
]
_VERIFY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "도시의 현재 날씨를 조회한다.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "도시 이름"}},
                "required": ["city"],
            },
        },
    }
]

# verify 서브커맨드가 쓰는 고정 한국어 프롬프트. 매번 같은 입력이어야 출력 비교가 된다.
VERIFY_PROMPT = "임진왜란이 일어난 연도와 그 당시 조선의 왕을 알려주세요."
VERIFY_MAX_TOKENS = 16


# 공통 유틸
def _hr(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


def _fmt_bytes(n: Optional[float]) -> str:
    if n is None:
        return "n/a"
    v = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(v) < 1024 or unit == "TiB":
            return f"{v:,.2f} {unit}"
        v /= 1024
    return f"{v}"


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """텐서 데이터를 불러오지 않고 safetensors 헤더만 읽는다."""
    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) < 8:
            raise ValueError(f"safetensors 헤더를 읽을 수 없다(파일이 너무 짧다): {path}")
        n = struct.unpack("<Q", raw)[0]
        if n <= 0 or n > 256 * 1024 * 1024:
            raise ValueError(f"safetensors 헤더 길이가 이상하다({n}): {path}")
        return json.loads(fh.read(n).decode("utf-8"))


def resolve_model_id(ckpt: Path) -> tuple[str, str]:
    """체크포인트 meta.json 에서 베이스 모델 ID를 읽는다. 실패하면 기본값.

    반환: (model_id, 출처 설명)
    """
    meta_path = ckpt / "meta.json"
    if not meta_path.is_file():
        return DEFAULT_MODEL_ID, f"meta.json 없음 ({meta_path}): 기본값 사용"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc: # noqa: BLE001
        return DEFAULT_MODEL_ID, f"meta.json 파싱 실패({exc}): 기본값 사용"
    mid = meta.get("exaone_id")
    if not mid:
        return DEFAULT_MODEL_ID, f"meta.json 에 exaone_id 없음: 기본값 사용"
    return str(mid), f"{meta_path}의 exaone_id"


def hub_cache_root() -> Path:
    """HF 허브 캐시 루트. 환경변수를 존중한다."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def resolve_snapshot(model_id: str) -> tuple[Optional[Path], str]:
    """필요한 파일이 있는 최신 로컬 스냅샷을 찾는다."""
    root = hub_cache_root()
    repo_dir = root / ("models--" + model_id.replace("/", "--"))
    if not repo_dir.is_dir():
        return None, f"허브 캐시에 없음: {repo_dir}"
    snaps_dir = repo_dir / "snapshots"
    if not snaps_dir.is_dir():
        return None, f"snapshots/ 없음: {snaps_dir}"

    candidates: list[tuple[float, Path]] = []
    rejected: list[str] = []
    for snap in sorted(snaps_dir.iterdir()):
        if not snap.is_dir():
            continue
        has_config = (snap / "config.json").is_file()
        has_weights = bool(list(snap.glob("model*.safetensors")))
        if has_config and has_weights:
            candidates.append((snap.stat().st_mtime, snap))
        else:
            missing = []
            if not has_config:
                missing.append("config.json")
            if not has_weights:
                missing.append("model*.safetensors")
            rejected.append(f"{snap.name}({','.join(missing)} 없음)")

    if not candidates:
        note = f"쓸 만한 스냅샷 없음: {snaps_dir}"
        if rejected:
            note += " | 후보 탈락: " + ", ".join(rejected)
        return None, note

    candidates.sort()
    chosen = candidates[-1][1]
    note = f"스냅샷 {len(candidates)}개 중 최신 선택 (rev={chosen.name})"
    if rejected:
        note += f" | 탈락 {len(rejected)}개"
    return chosen, note


def snapshot_weight_bytes(snapshot: Path) -> int:
    """스냅샷 안 safetensors 총 바이트. 심볼릭 링크를 따라간다."""
    total = 0
    for p in snapshot.glob("model*.safetensors"):
        try:
            total += p.resolve().stat().st_size
        except OSError:
            pass
    return total


def estimate_output_bytes(src_weight_bytes: int, bits: Optional[int], group_size: int) -> int:
    """가중치 수와 양자화 설정으로 출력 크기를 추정."""
    n_params = max(0, src_weight_bytes // 2)
    if bits is None:
        return src_weight_bytes
    bytes_per_param = bits / 8.0 + 4.0 / max(1, group_size)
    return int(n_params * bytes_per_param * 1.15)


def disk_guard(
    target_dir: Path,
    projected_bytes: int,
    min_free_gb: float,
    *,
    label: str,
) -> tuple[bool, str]:
    """변환 후 남을 공간을 계산하고, 하한 아래면 거부한다.

    반환: (진행해도 되는가, 사람이 읽을 설명)
    """
    probe = target_dir
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    free_before = usage.free
    free_after = free_before - projected_bytes
    min_free = int(min_free_gb * GIB)

    lines = [
        f"  마운트 지점 기준 : {probe}",
        f"  변환 전 여유     : {_fmt_bytes(free_before)}",
        f"  예상 출력 크기   : {_fmt_bytes(projected_bytes)}  ({label}, 추정치 +15% 여유 포함)",
        f"  변환 후 여유(추정): {_fmt_bytes(free_after)}",
        f"  하한(--min-free-gb): {_fmt_bytes(min_free)}",
    ]
    if free_after < min_free:
        lines.append(
            f"  판정: 거부: 변환 후 여유 {_fmt_bytes(free_after)} < 하한 {_fmt_bytes(min_free)}. "
            "디스크가 꽉 차면 macOS 스왑 파일이 못 늘어나 시스템 전체가 멈춘다."
        )
        return False, "\n".join(lines)
    lines.append("  판정: 통과")
    return True, "\n".join(lines)


# 토크나이저 복원과 검증
def restore_tokenizer_files(snapshot: Path, out_dir: Path) -> list[str]:
    """원본 스냅샷의 토크나이저 파일을 변환 결과 위에 덮어쓴다."""
    restored: list[str] = []
    for name in TOKENIZER_FILES:
        src = snapshot / name
        if not src.is_file():
            continue
        dst = out_dir / name
        before = dst.stat().st_size if dst.is_file() else 0
        shutil.copyfile(src.resolve(), dst) # resolve(): 허브 캐시는 심볼릭 링크다
        after = dst.stat().st_size
        restored.append(f"{name}: {before:,} B: {after:,} B")
    return restored


def verify_tokenizer(model_dir: Path) -> list[str]:
    """토크나이저만 로드해서 두 가지를 확인한다. 모델 가중치는 로드하지 않는다.

    문제 목록을 반환한다(비어 있으면 정상).
    """
    problems: list[str] = []
    try:
        from transformers import AutoTokenizer # noqa: PLC0415
    except ImportError as exc:
        return [f"transformers import 실패: 토크나이저 검증 불가: {exc}"]

    try:
        tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    except Exception as exc: # noqa: BLE001
        return [f"토크나이저 로드 실패: {type(exc).__name__}: {exc}"]

    # 종료 토큰 id를 확인한다
    eot = tok.convert_tokens_to_ids("[|endofturn|]")
    unk = getattr(tok, "unk_token_id", None)
    if eot is None or (unk is not None and eot == unk):
        problems.append(f"[|endofturn|] 가 실제 토큰으로 풀리지 않는다 (id={eot}, unk={unk})")
    else:
        print(f"  OK   [|endofturn|]: id {eot}")
        # 설정의 eos_token_id와 실제 토큰 id를 비교한다
        cfg_path = model_dir / "config.json"
        if cfg_path.is_file():
            try:
                cfg_eos = json.loads(cfg_path.read_text(encoding="utf-8")).get("eos_token_id")
            except Exception: # noqa: BLE001
                cfg_eos = None
            if isinstance(cfg_eos, int) and cfg_eos != eot:
                problems.append(
                    f"config.json eos_token_id={cfg_eos} 인데 [|endofturn|]={eot}: 불일치"
                )
            elif isinstance(cfg_eos, int):
                print(f"  OK   config.json eos_token_id={cfg_eos} 와 일치")

    # 도구 호출 템플릿을 확인한다
    try:
        prompt = tok.apply_chat_template(
            _VERIFY_MESSAGES,
            tools=_VERIFY_TOOLS,
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception as exc: # noqa: BLE001
        problems.append(f"apply_chat_template(tools=...) 실패: {type(exc).__name__}: {exc}")
        return problems

    if "<tool_call>" not in prompt:
        problems.append(
            "apply_chat_template(tools=...) 결과에 <tool_call> 이 없다: "
            "chat_template.jinja 가 유실됐거나 도구 분기를 타지 않았다"
        )
    else:
        print(f"  OK   apply_chat_template(tools=...) 에 <tool_call> 포함 ({len(prompt):,}자)")

    if "get_weather" not in prompt:
        problems.append("도구 이름(get_weather)이 템플릿 출력에 없다: 도구가 전달되지 않았다")

    return problems


# convert-base
def out_dir_for_bits(bits: Optional[int], ckpt: str | os.PathLike | None = None) -> Path:
    """<ckpt>/mlx/exaone4-1.2b-<tag>. 이름은 runtime._QUANT_DIRS 와 맞아야 한다."""
    tag = "bf16" if bits is None else f"{bits}bit"
    return mlx_root(ckpt) / f"exaone4-1.2b-{tag}"


def cmd_convert_base(args) -> int:
    bits = None if args.bits == "none" else int(args.bits)
    _hr(f"convert-base: bits={args.bits}" + ("  [DRY RUN]" if args.dry_run else ""))

    ckpt = Path(args.ckpt).expanduser()
    model_id, id_src = resolve_model_id(ckpt)
    print(f"모델 ID        : {model_id}")
    print(f"  출처         : {id_src}")

    snapshot, snap_note = resolve_snapshot(model_id)
    print(f"허브 캐시 루트 : {hub_cache_root()}")
    if snapshot is None:
        print(f"로컬 스냅샷    : 없음: {snap_note}")
        if not args.allow_download:
            print(
                "\n거부: 로컬 스냅샷을 못 찾았다. 저장소 ID를 그대로 넘기면 2.4 GiB를 다시\n"
                "      받게 되고, 이 기계는 그럴 여유가 없다. 정말 받으려면\n"
                "      --allow-download 를 명시하라."
            )
            return 1
        source = model_id
        src_weight_bytes = 2_600_000_000 # 다운로드 전이라 실측 불가. 알려진 대략치.
        print("경고: --allow-download 지정됨: 허브에서 새로 받을 수 있다.")
    else:
        source = str(snapshot)
        src_weight_bytes = snapshot_weight_bytes(snapshot)
        print(f"로컬 스냅샷    : {snapshot}")
        print(f"  선택 근거    : {snap_note}")
        print(f"  가중치 크기  : {_fmt_bytes(src_weight_bytes)}")
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                     "chat_template.jinja", "special_tokens_map.json"):
            p = snapshot / name
            mark = "OK  " if p.is_file() else "MISS"
            size = f"{p.resolve().stat().st_size:>12,} B" if p.is_file() else " " * 14
            print(f"  {mark} {name:26s}{size}")

    out_dir = Path(args.out).expanduser() if args.out else out_dir_for_bits(bits, ckpt)
    print(f"\n출력 경로      : {out_dir}")
    if out_dir.exists():
        print(
            "  주의: 이미 존재한다. mlx_lm.convert 는 기존 경로에 쓰기를 거부한다.\n"
            "        지우고 다시 만들거나 --out 으로 다른 경로를 주어야 한다."
        )

    projected = estimate_output_bytes(src_weight_bytes, bits, args.group_size)
    label = "bf16" if bits is None else f"{bits}bit/g{args.group_size}/{args.q_mode}"
    print("\n디스크 가드")
    ok, report = disk_guard(out_dir, projected, args.min_free_gb, label=label)
    print(report)

    print("\n변환 인자 (mlx_lm.convert.convert)")
    conv_kwargs: dict[str, Any] = {
        "hf_path": source,
        "mlx_path": str(out_dir),
        "quantize": bits is not None,
        "q_bits": bits,
        "q_group_size": args.group_size if bits is not None else None,
        "q_mode": args.q_mode,
        "dtype": args.dtype,
        "quant_predicate": args.quant_predicate,
    }
    for k, v in conv_kwargs.items():
        print(f"  {k:16s} = {v!r}")

    if args.dry_run:
        print("\nDRY RUN: 여기서 멈춘다. 가중치는 읽지도 쓰지도 않았다.")
        print("다음 단계 (실제 변환):")
        print(f"  python -m kava.serving.klava_mlx.convert convert-base --bits {args.bits}")
        # dry-run은 거부 판정을 출력해도 성공으로 끝낸다
        return 0

    if not ok:
        print("\n중단: 디스크 가드 거부. --min-free-gb 를 낮추려면 그 위험을 이해하고 하라.")
        return 2
    if out_dir.exists():
        print(f"\n중단: 출력 경로가 이미 있다: {out_dir}")
        return 2

    out_dir.parent.mkdir(parents=True, exist_ok=True)

    from mlx_lm.convert import convert # noqa: PLC0415  (import 시점 부작용 회피)

    print("\n[convert] 시작")
    convert(**conv_kwargs)
    print("[convert] 완료")

    actual = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    print(f"실제 출력 크기 : {_fmt_bytes(actual)} (추정 {_fmt_bytes(projected)})")

    if snapshot is not None:
        print("\n토크나이저 원본 복원")
        for line in restore_tokenizer_files(snapshot, out_dir):
            print(f"  복원 {line}")
    else:
        print("\n토크나이저 복원 건너뜀: 로컬 스냅샷 경로를 모른다(허브에서 직접 받았다)")

    print("\n토크나이저 검증")
    problems = verify_tokenizer(out_dir)
    if problems:
        print("검증 실패:")
        for p in problems:
            print("  -", p)
        return 3
    print("\n완료:", out_dir)
    return 0


# PEFT와 MLX의 행렬 배치가 달라 LoRA 가중치를 전치한다
PEFT_PREFIX = "base_model.model.model."
LORA_KEY_RE_SRC = r"^model\.layers\.(\d+)\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj\.lora_[ab]$"

# linear_to_lora_layers는 TransformerBlock 내부의 상대 경로를 사용한다
MLX_LORA_KEYS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def peft_key_to_mlx(key: str) -> Optional[tuple[str, str]]:
    """PEFT 텐서 키를 mlx-lm 키로 바꾼다. 반환: (mlx_key, 'a'|'b') 또는 None."""
    if not key.startswith(PEFT_PREFIX):
        return None
    tail = key[len(PEFT_PREFIX):] # model.layers.N.self_attn.q_proj.lora_A.weight
    if tail.endswith(".lora_A.weight"):
        which = "a"
        base = tail[: -len(".lora_A.weight")]
    elif tail.endswith(".lora_B.weight"):
        which = "b"
        base = tail[: -len(".lora_B.weight")]
    else:
        return None
    return f"model.{base}.lora_{which}", which


def _lora_scale_from_config(cfg: dict) -> float:
    """PEFT 설정에서 MLX LoRA scale을 계산."""
    if cfg.get("use_rslora"):
        raise ValueError("use_rslora=true 인 어댑터는 지원하지 않는다 (scale 식이 alpha/sqrt(r) 이다)")
    if cfg.get("use_dora"):
        raise ValueError("use_dora=true 인 어댑터는 지원하지 않는다 (fine_tune_type=dora 경로가 필요하다)")
    r = cfg.get("r")
    alpha = cfg.get("lora_alpha")
    if not r or alpha is None:
        raise ValueError(f"adapter_config.json에 r과 lora_alpha가 없다: r={r!r} alpha={alpha!r}")
    if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise ValueError("rank_pattern과 alpha_pattern이 비어 있지 않다: 레이어별 scale이 필요하다")
    return float(alpha) / float(r)


def cmd_convert_lora(args) -> int:
    _hr("convert-lora: PEFT: mlx-lm 어댑터" + ("  [DRY RUN]" if args.dry_run else ""))

    ckpt = Path(args.ckpt).expanduser()
    # --adapter 를 안 주면 체크포인트 안의 PEFT 어댑터를 쓴다(같은 체크포인트에서 나온 것).
    adapter_dir = Path(args.adapter).expanduser() if args.adapter else ckpt / "adapter"
    cfg_path = adapter_dir / "adapter_config.json"
    w_path = adapter_dir / "adapter_model.safetensors"
    out_dir = Path(args.out).expanduser() if args.out else mlx_root(ckpt) / "lora-mlx"

    print(f"입력 어댑터    : {adapter_dir}")
    for p in (cfg_path, w_path):
        mark = "OK  " if p.is_file() else "MISS"
        size = f"{p.stat().st_size:>14,} B" if p.is_file() else ""
        print(f"  {mark} {p.name:28s}{size}")
    if not cfg_path.is_file() or not w_path.is_file():
        print("\n중단: 어댑터 파일이 없다.")
        return 2

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    try:
        scale = _lora_scale_from_config(cfg)
    except ValueError as exc:
        print(f"\n중단: {exc}")
        return 2

    r = int(cfg["r"])
    alpha = int(cfg["lora_alpha"])
    print(f"\nLoRA 하이퍼파라미터 (adapter_config.json 에서 읽음)")
    print(f"  r            = {r}")
    print(f"  lora_alpha   = {alpha}")
    print(f"  scale        = alpha / r = {alpha} / {r} = {scale}")
    print(f"  lora_dropout = {cfg.get('lora_dropout')} (추론 전용이라 0.0 으로 내보낸다)")
    print(f"  target_modules = {sorted(cfg.get('target_modules') or [])}")

    # 체크포인트에서 확인한 scale과 일치하는지 검사한다
    if abs(scale - 2.0) > 1e-9:
        print(
            f"\n중단: scale 이 2.0 이 아니다 (={scale}). 사전 조사에서 이 체크포인트는 "
            "alpha=128, r=64: scale=2.0 으로 확인됐다. 값이 달라졌다면 어댑터가 바뀐 "
            "것이므로, 확인 없이 진행하면 안 된다."
        )
        return 2
    print("  검증: scale == 2.0  (사전 조사 값과 일치)")

    if cfg.get("modules_to_save"):
        print(f"\n중단: modules_to_save={cfg['modules_to_save']}: mlx 어댑터 포맷으로 못 옮긴다.")
        return 2
    if cfg.get("bias", "none") != "none" or cfg.get("lora_bias"):
        print(f"\n중단: bias 가 학습됐다 (bias={cfg.get('bias')}, lora_bias={cfg.get('lora_bias')}).")
        return 2

    # 텐서 헤더로 변환 계획을 확인한다
    hdr = read_safetensors_header(w_path)
    src_keys = [k for k in hdr if k != "__metadata__"]
    print(f"\n소스 텐서      : {len(src_keys)}개")

    mapped: dict[str, tuple[str, str, list[int], str]] = {}
    unmapped: list[str] = []
    for k in src_keys:
        conv = peft_key_to_mlx(k)
        if conv is None:
            unmapped.append(k)
            continue
        mlx_key, which = conv
        shape = list(hdr[k]["shape"])
        mapped[k] = (mlx_key, which, shape, hdr[k]["dtype"])

    if unmapped:
        print(f"  매핑 실패 {len(unmapped)}개:")
        for k in unmapped[:10]:
            print("   -", k)
        print("\n중단: 매핑되지 않는 키가 있어 어댑터를 일부만 적용할 수 없습니다.")
        return 2

# 키 형태와 shape 검증
    import re # noqa: PLC0415

    key_re = re.compile(LORA_KEY_RE_SRC)
    problems: list[str] = []
    layer_ids: set[int] = set()
    for src, (mlx_key, which, shape, dtype) in mapped.items():
        m = key_re.match(mlx_key)
        if not m:
            problems.append(f"{mlx_key}: mlx 키 패턴에 맞지 않는다")
            continue
        layer_ids.add(int(m.group(1)))
        if len(shape) != 2:
            problems.append(f"{src}: 2차원이 아니다 shape={shape}")
            continue
        # PEFT: A=(r, in), B=(out, r): 전치 후 a=(in, r), b=(r, out)
        if which == "a" and shape[0] != r:
            problems.append(f"{src}: lora_A 의 첫 차원이 r({r})이 아니다 shape={shape}")
        if which == "b" and shape[1] != r:
            problems.append(f"{src}: lora_B 의 둘째 차원이 r({r})이 아니다 shape={shape}")

    n_layers = (max(layer_ids) + 1) if layer_ids else 0
    expected = n_layers * len(MLX_LORA_KEYS) * 2
    print(f"  레이어 인덱스  : 0 ~ {max(layer_ids) if layer_ids else -1} (총 {len(layer_ids)}개)")
    print(f"  기대 텐서 수   : {n_layers} 레이어 x {len(MLX_LORA_KEYS)} 모듈 x 2 = {expected}")
    if len(mapped) != expected:
        problems.append(f"텐서 수 불일치: 실제 {len(mapped)} != 기대 {expected}")
    if len(layer_ids) != n_layers:
        problems.append(f"레이어 인덱스에 구멍이 있다: {sorted(set(range(n_layers)) - layer_ids)}")

    if problems:
        print("\n검증 문제:")
        for p in problems:
            print("  -", p)
        print("\n중단.")
        return 2
    print("  검증: 키 패턴과 shape 및 개수 모두 정상")

    adapter_config = {
        "fine_tune_type": "lora",
        "num_layers": n_layers,
        "lora_parameters": {
            "rank": r,
            "scale": scale,
            "dropout": 0.0,
            "keys": MLX_LORA_KEYS,
        },
        # 아래는 mlx-lm 이 안 읽는 기록용 필드. 나중에 이 파일만 보고도 출처를 알 수 있게 한다.
        "_source": {
            "peft_adapter": str(adapter_dir),
            "peft_version": cfg.get("peft_version"),
            "base_model_name_or_path": cfg.get("base_model_name_or_path"),
            "lora_alpha": alpha,
            "r": r,
            "lora_dropout_original": cfg.get("lora_dropout"),
            "note": (
                "scale 은 adapter_config.json 의 lora_alpha/r 에서 계산했다. "
                "mlx 의 LoRALinear.from_base 기본값 20.0 을 쓰면 10배 틀린다. "
                "dropout 은 추론 전용이라 0.0 으로 내보냈다(load_adapters 가 model.eval() 을 "
                "부르므로 어차피 비활성이지만, 명시가 안전하다)."
            ),
        },
    }

    print(f"\n출력 경로      : {out_dir}")
    print("  adapters.safetensors      (텐서 %d개, dtype=%s)" % (len(mapped), args.lora_dtype))
    print("  adapter_config.json:")
    for line in json.dumps(
        {k: v for k, v in adapter_config.items() if k != "_source"},
        indent=2, ensure_ascii=False,
    ).splitlines():
        print("    " + line)

    if args.dry_run:
        print("\n샘플 키 매핑 (앞 4개)")
        for src in sorted(mapped)[:4]:
            mlx_key, which, shape, dtype = mapped[src]
            out_shape = [shape[1], shape[0]]
            print(f"  {src}")
            print(f"   : {mlx_key}   {shape} 를 전치해 {out_shape}  ({dtype}: {args.lora_dtype})")
        print("\nDRY RUN: 여기서 멈춘다. safetensors 헤더만 읽었고 텐서 데이터는 안 읽었다.")
        print("다음 단계 (실제 변환):")
        print("  python -m kava.serving.klava_mlx.convert convert-lora")
        return 0

    # 가중치 변환
    import mlx.core as mx # noqa: PLC0415

    dtype_map = {"float32": mx.float32, "bfloat16": mx.bfloat16, "float16": mx.float16}
    target_dtype = dtype_map[args.lora_dtype]

    print("\n[변환] safetensors 로드 중...")
    src_tensors = mx.load(str(w_path))
    out_tensors: dict[str, Any] = {}
    for src, (mlx_key, which, shape, _dtype) in mapped.items():
        # .T 는 mlx 에서 lazy view 다. 저장 시점에 실제로 평가된다.
        out_tensors[mlx_key] = src_tensors[src].T.astype(target_dtype)
    mx.eval(out_tensors)
    del src_tensors

    out_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out_dir / "adapters.safetensors"), out_tensors)
    (out_dir / "adapter_config.json").write_text(
        json.dumps(adapter_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[변환] 완료: {out_dir}")

    # 변환 결과 검증
    print("\n되읽기 검증")
    out_hdr = read_safetensors_header(out_dir / "adapters.safetensors")
    out_keys = [k for k in out_hdr if k != "__metadata__"]
    v_problems: list[str] = []
    if len(out_keys) != expected:
        v_problems.append(f"텐서 수 {len(out_keys)} != 기대 {expected}")
    for k in out_keys:
        if not key_re.match(k):
            v_problems.append(f"{k}: mlx 키 패턴 위반")
            continue
        # 모듈 경로가 mlx 가 실제로 만드는 경로 집합 안에 있는지
        rel = k.split(".", 3)[3].rsplit(".", 1)[0] # self_attn.q_proj
        if rel not in MLX_LORA_KEYS:
            v_problems.append(f"{k}: 모듈 경로 {rel!r} 가 lora_parameters.keys 에 없다")
        shape = out_hdr[k]["shape"]
        if k.endswith(".lora_a") and shape[1] != r:
            v_problems.append(f"{k}: lora_a 둘째 차원이 r 이 아니다 {shape}")
        if k.endswith(".lora_b") and shape[0] != r:
            v_problems.append(f"{k}: lora_b 첫째 차원이 r 이 아니다 {shape}")
    if v_problems:
        print("검증 실패:")
        for p in v_problems[:20]:
            print("  -", p)
        return 3
    print(f"  OK   텐서 {len(out_keys)}개, 키 패턴과 모듈 경로 및 shape 정상")
    size = (out_dir / "adapters.safetensors").stat().st_size
    print(f"  OK   adapters.safetensors {_fmt_bytes(size)}")
    print("\n사용법: mlx_lm.load(model_dir, adapter_path=%r)" % str(out_dir))
    return 0


# verify
def cmd_verify(args) -> int:
    ckpt = Path(args.ckpt).expanduser()
    # --model 을 안 주면 서버 기본값(bf16)을 검증한다. 서버가 실제로 읽는 그 경로다.
    model_dir = Path(args.model).expanduser() if args.model else out_dir_for_bits(None, ckpt)
    if not model_dir.is_absolute():
        model_dir = (REPO_ROOT / model_dir).resolve()

    _hr(f"verify: {model_dir.name}" + ("  [DRY RUN]" if args.dry_run else ""))

    print(f"모델 디렉터리  : {model_dir}")
    problems: list[str] = []
    if not model_dir.is_dir():
        problems.append(f"디렉터리 없음: {model_dir}")
    else:
        weights = sorted(model_dir.glob("*.safetensors"))
        for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
            p = model_dir / name
            mark = "OK  " if p.is_file() else "MISS"
            print(f"  {mark} {name}")
            if not p.is_file():
                problems.append(f"파일 없음: {name}")
        print(f"  {'OK  ' if weights else 'MISS'} *.safetensors  ({len(weights)}개, "
              f"{_fmt_bytes(sum(p.stat().st_size for p in weights))})")
        if not weights:
            problems.append("safetensors 가중치 없음")
        cfg_p = model_dir / "config.json"
        if cfg_p.is_file():
            cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
            q = cfg.get("quantization")
            print(f"  model_type   = {cfg.get('model_type')}")
            print(f"  eos_token_id = {cfg.get('eos_token_id')}")
            print(f"  quantization = {q if q else '없음 (비양자화)'}")
            if cfg.get("model_type") != "exaone4":
                problems.append(f"model_type 이 exaone4 가 아니다: {cfg.get('model_type')}")

    # --adapter 를 안 주면 서버 기본 어댑터를 검증한다(--no-adapter 로 끌 수 있다).
    ad_arg = args.adapter if args.adapter else (None if args.no_adapter else str(mlx_root(ckpt) / "lora-mlx"))
    if ad_arg:
        ad = Path(ad_arg).expanduser()
        if not ad.is_absolute():
            ad = (REPO_ROOT / ad).resolve()
        print(f"어댑터         : {ad}")
        for name in ("adapter_config.json", "adapters.safetensors"):
            p = ad / name
            print(f"  {'OK  ' if p.is_file() else 'MISS'} {name}")
            if not p.is_file():
                problems.append(f"어댑터 파일 없음: {p}")

    print(f"\n생성 계획      : greedy {VERIFY_MAX_TOKENS} 토큰")
    print(f"  프롬프트     : {VERIFY_PROMPT}")

    if args.dry_run:
        if problems:
            print("\n확인된 문제:")
            for p in problems:
                print("  -", p)
        print("\nDRY RUN: 여기서 멈춘다. 모델을 로드하지 않았다.")
        return 0

    if problems:
        print("\n중단: 로드 전 검사에서 문제가 있다.")
        for p in problems:
            print("  -", p)
        return 2

    import mlx.core as mx # noqa: PLC0415
    from mlx_lm import load, stream_generate # noqa: PLC0415

    mx.reset_peak_memory()
    print("\n[load] ...")
    model, tokenizer = load(str(model_dir), adapter_path=str(ad) if ad_arg else None)
    print(f"[load] 완료. eos_token_ids={sorted(tokenizer.eos_token_ids)}")

    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": VERIFY_PROMPT}],
        add_generation_prompt=True,
        tokenize=True,
    )
    print(f"[gen] 프롬프트 {len(prompt_ids)} 토큰, greedy {VERIFY_MAX_TOKENS} 토큰 생성")

    pieces: list[str] = []
    ids: list[int] = []
    for resp in stream_generate(model, tokenizer, prompt_ids, max_tokens=VERIFY_MAX_TOKENS):
        pieces.append(resp.text)
        ids.append(resp.token)
    text = "".join(pieces)

    print("\n--- 생성 결과 ---")
    print(text)
    print("--- 끝 ---")
    print(f"토큰 {len(ids)}개: {ids}")
    print(f"mx.get_peak_memory() = {mx.get_peak_memory():,} B ({_fmt_bytes(mx.get_peak_memory())})")
    print(f"mx.get_active_memory() = {_fmt_bytes(mx.get_active_memory())}")
    print(f"mx.get_cache_memory()  = {_fmt_bytes(mx.get_cache_memory())}")

    if not text.strip():
        print("\n실패: 생성 결과가 비어 있다.")
        return 3
    return 0


# CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m kava.serving.klava_mlx.convert",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "MLX 모델 아티팩트 준비 (<ckpt_dir>/mlx/ 아래에 만든다).\n\n"
            f"현재 기본 ckpt : {DEFAULT_CKPT}\n"
            f"현재 기본 출력 : {mlx_root(DEFAULT_CKPT)}\n\n"
            "예:\n"
            "  python -m kava.serving.klava_mlx.convert convert-base --bits none --dry-run\n"
        ),
        epilog=(
            "모든 서브커맨드에 --dry-run 이 있다. 디스크가 빠듯하므로(여유 15 GB 안팎)\n"
            "실제 변환 전에 --dry-run으로 예상 크기와 디스크 상태를 확인한다.\n"
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def _common(sp):
        sp.add_argument("--dry-run", action="store_true",
                        help="무엇을 할지만 출력하고 가중치는 건드리지 않은 채 종료한다(exit 0).")
        sp.add_argument("--min-free-gb", type=float, default=6.0,
                        help="변환 후 남아 있어야 하는 최소 여유 공간(GiB). 기본 6.0. "
                             "이 아래로 내려가면 변환을 거부한다.")
        return sp

    # 베이스 모델 검증
    cb = _common(sub.add_parser(
        "convert-base",
        help="베이스 EXAONE 을 MLX 포맷으로 변환 (선택적으로 양자화)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="베이스 EXAONE 을 <ckpt>/mlx/exaone4-1.2b-{bf16,8bit,6bit,4bit}/ 로 변환한다.",
    ))
    cb.add_argument("--bits", choices=["none", "8", "6", "4"], default="none",
                    help="none=bf16 비양자화(기본). 8, 6, 4는 affine 양자화 비트 수. "
                         "mlx_lm은 2와 3도 받지만 1.2B 모델에서 품질이 낮아 제외했다.")
    cb.add_argument("--group-size", type=int, default=64,
                    help="양자화 그룹 크기 (mlx_lm 기본 64). 작을수록 정확하고 파일이 커진다.")
    cb.add_argument("--q-mode", choices=["affine", "mxfp4", "nvfp4", "mxfp8"], default="affine",
                    help="양자화 방식. 기본 affine. mxfp*와 nvfp*는 quant-predicate와 함께 못 쓴다.")
    cb.add_argument("--quant-predicate",
                    choices=["mixed_2_6", "mixed_3_4", "mixed_3_6", "mixed_4_6"], default=None,
                    help="레이어별 혼합 비트 정책 (affine 에서만 동작).")
    cb.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16",
                    help="비양자화 가중치 dtype. 기본 bfloat16: torch 베이스라인이 bfloat16 이라 맞춘 것.")
    cb.add_argument("--ckpt", default=str(DEFAULT_CKPT),
                    help="meta.json 에서 exaone_id 를 읽고, 출력 경로도 여기서 파생시킨다. "
                         "기본값은 KLAVA_CKPT 또는 저장소 기본 체크포인트.")
    cb.add_argument("--out", default=None,
                    help="출력 경로 (기본: <ckpt>/mlx/exaone4-1.2b-<tag>)")
    cb.add_argument("--allow-download", action="store_true",
                    help="로컬 스냅샷이 없을 때 허브에서 받는 것을 허용한다. 기본은 거부다"
                         "(2.4 GiB 를 다시 받을 여유가 없다).")
    cb.set_defaults(func=cmd_convert_base)

    # LoRA 검증
    cl = _common(sub.add_parser(
        "convert-lora",
        help="PEFT LoRA 어댑터를 mlx-lm 포맷으로 변환",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "PEFT 어댑터를 mlx-lm 어댑터로 옮긴다.\n"
            "lora_A와 lora_B를 전치하고 scale은 lora_alpha/r로 지정한다.\n"
            "     scale을 생략하면 MLX 기본값 20.0이 적용되어 결과가 달라진다."
        ),
    ))
    cl.add_argument("--ckpt", default=str(DEFAULT_CKPT),
                    help="PEFT 어댑터와 출력 경로를 파생시킬 체크포인트 디렉터리. "
                         "기본값은 KLAVA_CKPT 또는 저장소 기본 체크포인트.")
    cl.add_argument("--adapter", default=None, help="PEFT 어댑터 디렉터리 (기본: <ckpt>/adapter)")
    cl.add_argument("--out", default=None, help="출력 경로 (기본: <ckpt>/mlx/lora-mlx)")
    cl.add_argument("--lora-dtype", choices=["float32", "bfloat16", "float16"], default="float32",
                    help="어댑터 텐서 dtype. 기본 float32 = 학습된 값 그대로(바이트 단위 보존). "
                         "bfloat16 은 파일이 절반(243에서 122로 MB)이 되지만 값이 미세하게 바뀐다. "
                         "측정 하네스이므로 기본은 보수적으로 float32 로 둔다.")
    cl.set_defaults(func=cmd_convert_lora)

    # 전체 검증
    cv = _common(sub.add_parser(
        "verify",
        help="변환된 모델 스모크 테스트 (고정 한국어 프롬프트로 16토큰 greedy 생성)",
    ))
    cv.add_argument("--ckpt", default=str(DEFAULT_CKPT),
                    help="모델과 어댑터 기본 경로를 파생시킬 체크포인트 디렉터리.")
    cv.add_argument("--model", default=None,
                    help="검증할 MLX 모델 디렉터리 (기본: <ckpt>/mlx/exaone4-1.2b-bf16 "
                         "서버가 실제로 읽는 경로)")
    cv.add_argument("--adapter", default=None,
                    help="함께 얹을 mlx 어댑터 디렉터리 (기본: <ckpt>/mlx/lora-mlx)")
    cv.add_argument("--no-adapter", action="store_true",
                    help="어댑터 없이 베이스 LM 만 검증한다.")
    cv.set_defaults(func=cmd_verify)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
