"""KAVA 내부 오류 정의."""


# 서버 오류에는 내부 정보를 제외한 고정 문구를 사용한다
INTERNAL_DETAIL = "서버 내부 오류가 발생했습니다."

# 모델 서버 오류는 원인만 알리고 상류 응답은 제외한다
UPSTREAM_DETAIL = "모델 서버 호출이 실패했습니다. 잠시 후 다시 시도해 주세요."


class KAVAError(Exception):
    """모든 KAVA 관련 오류의 수퍼 클래스. code와 status_code로 HTTP 응답을 만듦."""
    code = "kava_error"
    status_code = 500 # 서버 에러

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail

class SessionNotFound(KAVAError):
    """없는 세션 조회 에러."""
    code = "session_not_found"
    status_code = 404 # Not found 에러

class SessionBusy(KAVAError):
    """같은 세션에서의 요청이 처리 중을 나타내는 에러."""
    code = "session_busy"
    status_code = 409 # 충돌 에러

class UpstreamFailure(KAVAError):
    """KLaVA, OCR, LM 실패 에러."""
    code = "upstream_failure"
    status_code = 502 # 상위 서버로 정상 응답을 받지 못함.

class AssetNotFound(KAVAError):
    """업로드되지 않은 이미지 선택 에러."""
    code = "asset_not_found"
    status_code = 404

class InvalidAsset(KAVAError):
    """이미지로 읽을 수 없거나 허용하지 않는 확장자의 업로드 에러."""
    code = "invalid_asset"
    status_code = 422 # 요청 형식은 맞지만 내용이 유효하지 않음

class AssetTooLarge(KAVAError):
    """허용 크기를 넘는 업로드 에러."""
    code = "asset_too_large"
    status_code = 413 # 본문이 너무 큼


def public_detail(exc: KAVAError) -> str:
    """오류 상태에 맞춰 외부에 공개할 문구를 반환."""
    if exc.status_code < 500:
        return exc.detail
    if exc.status_code == 502:
        return UPSTREAM_DETAIL
    return INTERNAL_DETAIL
