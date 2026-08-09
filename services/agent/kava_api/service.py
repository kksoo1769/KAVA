"""한 턴의 처리 흐름."""
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass

from kava_api.assets import AssetStore
from kava_api.domain import ConversationState, ImageRef, Message
from kava_api.errors import UpstreamFailure
from kava_api.graph_backend import Delta, GraphBackend, Progress, TurnResult
from kava_api.graph_state import ImgKind
from kava_api.store import ConversationStore


@dataclass(frozen=True, slots=True)
class TurnView:
    """한 턴의 요청에 대한 응답."""
    answer: str
    turn_count: int # 몇번째 턴인지
    history_reset: bool # 히스토리 리셋 여부
    img_id: str | None
    img_kind: ImgKind

@dataclass(frozen=True, slots=True)
class ConversationView:
    """대화 조회 응답. 내부 값(OCR 원문, 서버 파일 경로)은 담지 않는다."""
    session_id: str
    messages: list[Message]
    img_id: str | None
    img_kind: ImgKind
    has_ocr_cache: bool
    turn_count: int

class TurnService:
    """저장소 ConversationStore와 그래프를 한 턴에서 연결."""
    def __init__(
        self,
        store: ConversationStore,
        backend: GraphBackend,
        assets: AssetStore,
        max_turns: int,
        *,
        replay_size: int = 16,
    ) -> None:
        self._store = store
        self._backend = backend
        self._assets = assets
        self._max_turns = max_turns
        self._replies: OrderedDict[tuple[str, str], TurnView] = OrderedDict()
        self._replay_size = replay_size

    @property
    def assets(self) -> AssetStore:
        """업로드 엔드포인트가 쓰는 자산 저장소."""
        return self._assets

    async def stream_turn(
        self,
        session_id: str,
        question: str,
        img_id: str | None = None,
        *,
        clear_image: bool = False,
        request_id: str | None = None,
    ) -> AsyncIterator[Progress | Delta | TurnView]:
        """한 턴을 처리하면서 진행 상황과 답변 조각을 내고, 마지막에 결과를 낸다. 실패시 이전 상태로 되돌린다."""
        key = None if request_id is None else (session_id, request_id)
        if key is not None and key in self._replies:
            yield self._replies[key]
            return

        async with self._store.lease(session_id) as state:
            selected = None if img_id is None else self._assets.resolve(img_id)
            attaching_new_image = selected is not None and not state.is_same_image(selected)

            if attaching_new_image:
                img = selected
                history_reset = True
            elif clear_image:
                img = None
                history_reset = True
            elif selected is not None:
                # 붙어 있던 것과 같은 이미지를 다시 지목한 경우. 그대로 이어 쓴다
                img = selected
                history_reset = False
            else:
                # 이미지를 지목하지 않은 후속 턴. 붙어 있던 이미지를 이어 쓴다
                img = self._verify_attached(state)
                history_reset = False

            if history_reset:
                state.reset_context(img=img)

            view = None
            async for item in self._backend.stream_turn(state, question, img=img):
                if isinstance(item, (Progress, Delta)):
                    yield item
                    continue

                result: TurnResult = item
                state.img_kind = result.img_kind
                state.ocr_cache = result.ocr_cache
                state.append_turn(question, result.answer, max_turns=self._max_turns)

                view = TurnView(
                    answer=result.answer,
                    turn_count=state.turn_count,
                    history_reset=history_reset,
                    img_id=state.img.id if state.img is not None else None,
                    img_kind=state.img_kind,
                )
                yield view

            if view is None:
                raise UpstreamFailure("답변을 만들지 못했습니다.")

        # 같은 request_id 재요청에 실패한 턴을 성공처럼 되돌려준다.
        if key is not None:
            self._remember(key, view)

    async def run_turn(
        self,
        session_id: str,
        question: str,
        img_id: str | None = None,
        *,
        clear_image: bool = False,
        request_id: str | None = None,
    ) -> TurnView:
        """스트리밍이 없는 한 턴의 결과만 반환.

        stream_turn을 사용하되, 처리 중에 발생하는 Progress와 Delta를 무시하고 끝까지 처리 후 최종 TurnView만 확인해 반환한다."""
        view = None
        async for item in self.stream_turn(
            session_id, question, img_id, clear_image=clear_image, request_id=request_id
        ):
            if isinstance(item, TurnView):
                view = item
        if view is None:
            raise UpstreamFailure("답변을 만들지 못했습니다.")
        return view

    async def get_conversation(self, session_id: str) -> ConversationView:
        """대화를 반환하고, 없으면 SessionNotFound를 예외로 뱉음."""
        state = await self._store.get(session_id)
        return ConversationView(
            session_id=state.session_id,
            messages=list(state.messages),
            img_id=state.img.id if state.img is not None else None,
            img_kind=state.img_kind,
            has_ocr_cache=state.ocr_cache is not None, # 원문이 아니라 있고 없음만 알린다
            turn_count=state.turn_count,
        )

    async def delete_conversation(self, session_id: str) -> bool:
        """대화를 삭제하고, 삭제 여부를 반환."""
        return await self._store.delete(session_id)

    def _verify_attached(self, state: ConversationState) -> ImageRef | None:
        """이어 쓰는 이미지가 아직 디스크에 있는지 확인."""
        if state.img is None:
            return None
        return self._assets.resolve(state.img.id)

    def _remember(self, key: tuple[str, str], view: TurnView) -> None:
        """같은 request_id 재요청에 돌려줄 응답을 최근 것만 남겨 기억."""
        self._replies[key] = view
        while len(self._replies) > self._replay_size:
            self._replies.popitem(last=False)
