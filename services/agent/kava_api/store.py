"""세션별 대화를 메모리에 보관."""
import asyncio
import copy
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from kava_api.domain import ConversationState
from kava_api.errors import SessionBusy, SessionNotFound


class ConversationStore:
    """그래프에서 대신 대화를 관리하는 객체."""
    def __init__(self) -> None:
        self._sessions: dict[str, ConversationState] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @asynccontextmanager
    async def lease(self, session_id: str) -> AsyncGenerator[ConversationState]:
        """세션 복사본을 잠근 상태로 제공하고 정상 종료 시 저장."""
        lock = self._locks.get(session_id) # session_id가 있으면 반환
        if lock is None: # 없으면 Lock 생성 후 기록
            lock = self._locks[session_id] = asyncio.Lock()
        if lock.locked(): # 사용자가 실수로 두 번 눌렀을 때 거절하고, 이미 처리 중임을 전달.
            raise SessionBusy(f"세션 {session_id}가 이미 처리 중입니다.")

        try:
            async with lock: # 동시 사용 방지
                confirmed = self._sessions.get(session_id)
                working = copy.deepcopy(confirmed) if confirmed is not None else ConversationState(session_id=session_id)
                yield working # 예외 발생 시 자동 롤백

                self._sessions[session_id] = working # 커밋
        finally:
            if session_id not in self._sessions:
                self._locks.pop(session_id, None)

    async def get(self, session_id: str) -> ConversationState:
        """정상 확정된 대화를 복사해 반환하고, 없으면 SessionNotFound 예외를 뱉는다."""
        state = self._sessions.get(session_id)
        if state is None:
            raise SessionNotFound(f"세션 {session_id}가 존재하지 않습니다.")
        return copy.deepcopy(state)

    async def delete(self, session_id: str) -> bool:
        """존재하는 세션은 삭제하고, 삭제 유무 반환.

        처리 중인 세션은 지우지 않고 SessionBusy를 뱉는다."""
        lock = self._locks.get(session_id)
        if lock is not None and lock.locked(): # 처리 중인 세션이면
            raise SessionBusy(f"세션 {session_id}가 이미 처리 중입니다.")
        self._locks.pop(session_id, None)
        return self._sessions.pop(session_id, None) is not None
