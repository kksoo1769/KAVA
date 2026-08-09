"""torch와 MLX 추론 백엔드가 공유하는 경량 인터페이스."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class StreamResult:
    """스트림의 마지막에 전달되는 최종 결과."""

    value: str | dict


@runtime_checkable
class SharedRuntimeProtocol(Protocol):
    """두 추론 런타임이 제공하는 공통 메서드."""

    # 요청 큐
    queue: Any
    # 워커 스레드
    worker: Any

    def run(self, kind: str, payload: dict, timeout: float | None = ...) -> Any:
        """job 을 큐에 넣고 결과를 기다린다. vlm: str, chat: dict."""
        ...

    def stream(
        self, kind: str, payload: dict, timeout: float | None = ...
    ) -> Iterator[str | StreamResult]:
        """증분 텍스트를 전달한 뒤 StreamResult로 마무리."""
        ...

    def shutdown(self, timeout: float = ...) -> None:
        """워커에 종료 신호를 넣고 합류를 기다린다."""
        ...
