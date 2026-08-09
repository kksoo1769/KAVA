"use client";

import { ChangeEvent, KeyboardEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

import Markdown from "./components/Markdown";

type Role = "user" | "assistant";

type ChatMessage = {
  id: string;
  role: Role;
  content: string;
  imageName?: string;
  /** 지금 델타를 받아 자라고 있는 어시스턴트 메시지 표식. done이 오면 false로 내린다. */
  streaming?: boolean;
};

type SessionSummary = {
  id: string;
  title: string;
  updatedAt: number;
};

type ImageState = {
  imageId: string;
  fileName: string;
  previewUrl: string;
  ownsPreviewUrl?: boolean;
};

type SessionResponse = {
  session_id: string;
  turn_count: number;
  messages: Array<{ role: Role; content: string }>;
  active_image: { image_id: string; kind: string | null } | null;
  has_ocr_cache: boolean;
};

type TurnResponse = {
  session_id: string;
  answer: string;
  turn_count: number;
  history_reset: boolean;
  active_image: { image_id: string; kind: string | null } | null;
};

type ProblemDetail = {
  code?: string;
  detail?: string;
};

const API_BASE = (process.env.NEXT_PUBLIC_KAVA_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
const SESSION_STORAGE_KEY = "kava.local.sessions.v1";

async function fetchWithTimeout(input: RequestInfo | URL, init: RequestInit = {}, timeoutMs = 10_000) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } finally {
    window.clearTimeout(timer);
  }
}

function makeId(prefix: string) {
  return `${prefix}_${crypto.randomUUID()}`;
}

function assetUrl(imageId: string) {
  return `${API_BASE}/v1/assets/${encodeURIComponent(imageId)}`;
}

function newSession(): SessionSummary {
  return { id: makeId("session"), title: "새 대화", updatedAt: Date.now() };
}

function titleFrom(question: string) {
  const compact = question.trim().replace(/\s+/g, " ");
  return compact.length > 32 ? `${compact.slice(0, 32)}…` : compact;
}

function friendlyError(reason: unknown) {
  if (reason instanceof DOMException && reason.name === "AbortError") {
    return "KAVA 응답 시간이 초과되었습니다. 로컬 모델 상태를 확인하세요.";
  }
  if (reason instanceof TypeError && /fetch/i.test(reason.message)) {
    return `KAVA Agent API에 연결할 수 없습니다. ${API_BASE} 서버를 확인하세요.`;
  }
  return reason instanceof Error ? reason.message : "요청을 처리하지 못했습니다.";
}

async function readProblem(response: Response) {
  try {
    const body = (await response.json()) as ProblemDetail;
    return body.detail || `요청이 실패했습니다 (${response.status})`;
  } catch {
    return `요청이 실패했습니다 (${response.status})`;
  }
}

async function fetchSession(sessionId: string): Promise<SessionResponse | null> {
  const response = await fetchWithTimeout(`${API_BASE}/v1/sessions/${encodeURIComponent(sessionId)}`);
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(await readProblem(response));
  return response.json();
}

async function deleteRemoteSession(sessionId: string) {
  const response = await fetchWithTimeout(`${API_BASE}/v1/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
  if (!response.ok) throw new Error(await readProblem(response));
}

async function uploadAsset(file: File): Promise<string> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetchWithTimeout(`${API_BASE}/v1/assets`, { method: "POST", body: form }, 120_000);
  if (!response.ok) throw new Error(await readProblem(response));
  const body = (await response.json()) as { image_id?: string };
  if (!body.image_id) throw new Error("이미지 등록 응답에 image_id가 없습니다.");
  return body.image_id;
}

type SseFrame = { event: string; data: Record<string, unknown> };

/**
 * SSE 프레임 구분자는 빈 줄 하나다. 다만 개행이 LF일지 CRLF일지는 서버와 중간 프록시가
 * 정하므로 둘 다 받아야 한다. 프레임 분할과 줄 분할이 같은 규칙을 쓰도록 상수로 묶어 둔다.
 */
const FRAME_SEPARATOR = /\r?\n\r?\n/;
const LINE_SEPARATOR = /\r?\n/;

/** SSE 프레임 하나를 event 이름과 data(JSON)로 나눈다. 읽을 것이 없으면 null이다. */
function parseFrame(frame: string): SseFrame | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of frame.split(LINE_SEPARATOR)) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) {
      // SSE 스펙이 버리는 것은 "콜론 바로 뒤 공백 한 칸"뿐이다. trim()으로 양쪽을 다 깎으면
      // 토큰 델타의 선행 공백이 함께 사라지는데, 그 공백이 마크다운에서 리스트 들여쓰기와
      // 코드 블록(4칸 인덴트) 판정을 좌우한다. 즉 오류 없이 렌더 결과만 조용히 깨진다.
      const value = line.slice(5);
      dataLines.push(value.startsWith(" ") ? value.slice(1) : value);
    }
  }
  if (!dataLines.length) return null;
  try {
    return { event, data: JSON.parse(dataLines.join("\n")) as Record<string, unknown> };
  } catch {
    return null;
  }
}

/**
 * 한 턴을 스트리밍으로 처리한다. 진행 단계는 onProgress로, 토큰 조각은 onDelta로 알리고
 * 최종 답변을 반환한다. 누적은 호출부가 한다. 여기서 문자열을 모아 두지 않는 덕분에
 * 화면 갱신 방식(어느 메시지에 붙일지, 얼마나 자주 그릴지)을 호출부가 정할 수 있다.
 *
 * EventSource는 GET만 되고 본문을 못 보내므로 fetch로 직접 읽는다. chunk 경계는 SSE 프레임
 * 경계와 무관해서 `data:` 줄 하나가 두 chunk에 걸쳐 쪼개져 올 수 있다. 그래서 버퍼에 모아
 * 두고 빈 줄이 나온 데까지만 잘라 쓰고, 남은 조각은 다음 chunk와 이어 붙인다.
 *
 * done 이벤트의 answer가 최종 진실이다. 누적한 델타는 기다리는 동안 보여 주기 위한
 * 임시 표현일 뿐이고, done이 오면 answer로 통째로 교체한다. 백엔드가 툴 호출 태그를
 * 흘렸거나 델타 몇 개가 유실됐어도 턴이 끝나는 순간 스스로 교정되는 구조다.
 */
async function streamTurn(
  sessionId: string,
  payload: {
    request_id: string;
    question: string;
    image?: { image_id: string };
    clear_image?: boolean;
  },
  onProgress: (label: string) => void,
  onDelta: (text: string) => void,
): Promise<TurnResponse> {
  const controller = new AbortController();
  // 타임아웃은 스트림을 다 읽을 때까지 걸어 둔다. abort는 읽는 중에도 reader.read()를 깨운다.
  const timer = window.setTimeout(() => controller.abort(), 600_000);
  try {
    const response = await fetch(`${API_BASE}/v1/sessions/${encodeURIComponent(sessionId)}/turns/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(await readProblem(response));
    if (!response.body) throw new Error("스트리밍 응답을 읽을 수 없습니다.");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let result: TurnResponse | null = null;

    /**
     * 프레임 하나를 소비한다. done 프레임이면 최종 응답을 돌려주고, 그 밖에는 null이다.
     * 반환값으로 넘기는 이유는 클로저에서 바깥 변수를 대입하면 타입 좁히기가 어긋나서다.
     */
    const consume = (frame: string): TurnResponse | null => {
      const parsed = parseFrame(frame);
      if (!parsed) return null;
      if (parsed.event === "progress") onProgress(String(parsed.data.label ?? ""));
      else if (parsed.event === "delta") onDelta(String(parsed.data.text ?? ""));
      else if (parsed.event === "done") return parsed.data as unknown as TurnResponse;
      else if (parsed.event === "error") {
        // 상태 코드는 이미 200으로 나갔으므로 오류가 이벤트로 온다. 여기서 예외로
        // 바꿔 던져 기존 오류 처리 흐름(배너 표시, 질문 되돌리기)을 그대로 태운다.
        throw new Error(String(parsed.data.detail ?? "요청을 처리하지 못했습니다."));
      }
      // 그 밖의 이벤트 이름은 조용히 흘려보낸다. 백엔드가 나중에 새 이벤트를 추가해도
      // 구버전 프론트가 깨지지 않는 것이 SSE를 쓰는 이유 중 하나다.
      return null;
    };

    try {
      let streamEnded = false;
      while (!result && !streamEnded) {
        const { done, value } = await reader.read();
        if (done) {
          streamEnded = true;
          buffer += decoder.decode(); // 걸쳐 있던 멀티바이트 잔여분을 마지막으로 비운다
        } else {
          buffer += decoder.decode(value, { stream: true });
        }

        const frames = buffer.split(FRAME_SEPARATOR);
        // 스트림이 끝났으면 남은 조각도 완결된 프레임으로 본다. 서버가 마지막 프레임 뒤에
        // 빈 줄을 붙이지 않으면 done 프레임이 buffer에 갇힌 채 루프가 끝나는데, 그러면
        // 정상 응답인데도 "연결이 끊겼습니다"로 오인한다. 아직 읽는 중이면 마지막 조각은
        // 덜 온 프레임이므로 버퍼에 되돌려 다음 chunk와 이어 붙인다.
        buffer = streamEnded ? "" : (frames.pop() ?? "");
        for (const frame of frames) {
          const finished = consume(frame);
          if (finished) result = finished;
        }
      }
    } finally {
      await reader.cancel().catch(() => {});
    }

    // done도 error도 없이 끊긴 스트림은 성공이 아니다. 서버가 죽었거나 연결이 끊긴 경우다.
    if (!result) throw new Error("답변을 받지 못한 채 연결이 끊겼습니다. 다시 시도해 주세요.");
    return result;
  } finally {
    window.clearTimeout(timer);
  }
}

export default function Home() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [activeSessionId, setActiveSessionId] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [pendingImageUrl, setPendingImageUrl] = useState("");
  const [activeImage, setActiveImage] = useState<ImageState | null>(null);
  const [imagePanelOpen, setImagePanelOpen] = useState(false);
  const [clearImage, setClearImage] = useState(false);
  const [sending, setSending] = useState(false);
  const [progress, setProgress] = useState(""); // 지금 진행 중인 단계 문구. 빈 문자열이면 표시하지 않는다
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [backendState, setBackendState] = useState<"checking" | "ready" | "offline">("checking");
  const [error, setError] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pendingImageUrlRef = useRef("");
  const activeImageUrlRef = useRef("");
  const messagesRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messageCountRef = useRef(0); // 자동 스크롤에서 "새 메시지"와 "내용만 자람"을 구분한다
  const scrolledSessionRef = useRef(""); // 대화를 갈아탄 순간을 알아내 끝으로 점프시킨다

  const clearPendingImage = useCallback(() => {
    setPendingFile(null);
    setPendingImageUrl("");
    if (pendingImageUrlRef.current) {
      URL.revokeObjectURL(pendingImageUrlRef.current);
      pendingImageUrlRef.current = "";
    }
  }, []);

  const replaceActiveImage = useCallback((next: ImageState | null) => {
    if (activeImageUrlRef.current && activeImageUrlRef.current !== next?.previewUrl) {
      URL.revokeObjectURL(activeImageUrlRef.current);
    }
    activeImageUrlRef.current = next?.ownsPreviewUrl ? next.previewUrl : "";
    setActiveImage(next);
    if (!next) setImagePanelOpen(false);
  }, []);

  const persistSessions = useCallback((next: SessionSummary[]) => {
    setSessions(next);
    localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(next));
  }, []);

  const addSession = useCallback(() => {
    const session = newSession();
    const next = [session, ...sessions];
    persistSessions(next);
    setActiveSessionId(session.id);
    setMessages([]);
    replaceActiveImage(null);
    clearPendingImage();
    setClearImage(false);
    setError("");
    setSidebarOpen(false);
  }, [clearPendingImage, persistSessions, replaceActiveImage, sessions]);

  useEffect(() => () => {
    if (pendingImageUrlRef.current) URL.revokeObjectURL(pendingImageUrlRef.current);
    if (activeImageUrlRef.current) URL.revokeObjectURL(activeImageUrlRef.current);
  }, []);

  useEffect(() => {
    let saved: SessionSummary[] = [];
    try {
      saved = JSON.parse(localStorage.getItem(SESSION_STORAGE_KEY) ?? "[]");
    } catch {
      saved = [];
    }
    if (saved.length) {
      // Client-only navigation metadata is restored once after hydration.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setSessions(saved);
      setActiveSessionId(saved[0].id);
    } else {
      const first = newSession();
      persistSessions([first]);
      setActiveSessionId(first.id);
    }
  }, [persistSessions]);

  useEffect(() => {
    let cancelled = false;
    async function checkBackend() {
      try {
        const response = await fetchWithTimeout(`${API_BASE}/readyz`, {}, 3_000);
        if (!cancelled) setBackendState(response.ok ? "ready" : "offline");
      } catch {
        if (!cancelled) setBackendState("offline");
      }
    }
    checkBackend();
    const timer = window.setInterval(checkBackend, 10_000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (!activeSessionId || backendState === "checking") return;
    if (backendState === "offline") return;
    let cancelled = false;
    fetchSession(activeSessionId)
      .then((session) => {
        if (cancelled) return;
        setMessages(
          (session?.messages ?? []).map((message) => ({
            ...message,
            id: makeId("message"),
          })),
        );
        replaceActiveImage(
          session?.active_image
            ? {
                imageId: session.active_image.image_id,
                fileName: "연결된 이미지",
                previewUrl: assetUrl(session.active_image.image_id),
              }
            : null,
        );
        setClearImage(false);
      })
      .catch((reason: Error) => {
        if (!cancelled) setError(friendlyError(reason));
      })
    return () => {
      cancelled = true;
    };
  }, [activeSessionId, backendState, replaceActiveImage]);

  useEffect(() => {
    const previousCount = messageCountRef.current;
    const previousSession = scrolledSessionRef.current;
    messageCountRef.current = messages.length;
    scrolledSessionRef.current = activeSessionId;
    const isNewMessage = messages.length !== previousCount;

    // 다른 대화를 열었거나 빈 화면이 처음 채워진 순간은 가드를 건너뛰고 무조건 끝으로 붙인다.
    // 대화를 열면 최신 메시지가 보여야 하고, 이때는 "바닥 근처인가" 판정이 의미가 없다.
    const jumpToEnd = activeSessionId !== previousSession || (previousCount === 0 && messages.length > 0);

    if (!jumpToEnd) {
      // 실제로 스크롤되는 요소를 찾는다. 지금 레이아웃은 .messages에 overflow가 없어 문서가
      // 스크롤되지만, 나중에 .messages에 overflow-y를 주더라도 그대로 동작하게 둘 다 본다.
      const pane = messagesRef.current;
      const scroller = pane && pane.scrollHeight > pane.clientHeight + 1 ? pane : document.scrollingElement;
      if (!scroller) return;

      // 사용자가 위로 올려 과거를 읽고 있으면 끌어내리지 않는다. 바닥에서 이 거리 안쪽이면
      // "따라가는 중"으로 본다. 한 줄 높이보다 넉넉해야 마지막 줄이 자랄 때 판정이 흔들리지 않는다.
      const NEAR_BOTTOM_PX = 120;
      const distanceToBottom = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight;
      if (distanceToBottom > NEAR_BOTTOM_PX) return;
    }

    // 메시지 개수가 바뀐 순간(새 턴 시작, 첫 델타로 답변 버블 등장)만 smooth로 부드럽게 붙인다.
    // 델타로 내용만 자라는 동안 smooth를 쓰면 매 토큰마다 목표가 새로 잡혀 애니메이션이
    // 목적지에 닿지 못하고 계속 재시작하며 떨린다. 그래서 그때는 auto로 즉시 붙인다.
    // 대화를 새로 여는 점프도 애니메이션 없이 즉시 끝으로 보낸다.
    const behavior = !jumpToEnd && isNewMessage ? "smooth" : "auto";
    messagesEndRef.current?.scrollIntoView({ behavior });
  }, [activeSessionId, messages, sending]);

  const activeSession = useMemo(
    () => sessions.find((session) => session.id === activeSessionId),
    [activeSessionId, sessions],
  );

  function selectFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    if (pendingImageUrlRef.current) URL.revokeObjectURL(pendingImageUrlRef.current);
    const previewUrl = file ? URL.createObjectURL(file) : "";
    pendingImageUrlRef.current = previewUrl;
    setPendingFile(file);
    setPendingImageUrl(previewUrl);
    if (file) {
      setClearImage(false);
      setImagePanelOpen(true);
    }
    event.target.value = "";
  }

  async function sendMessage() {
    const trimmed = question.trim();
    if (!trimmed || !activeSessionId || sending) return;

    const optimisticUser: ChatMessage = {
      id: makeId("message"),
      role: "user",
      content: trimmed,
      imageName: pendingFile?.name,
    };
    // 델타를 이어붙일 자리를 미리 만들어 둔다. content가 빈 문자열인 동안에는 이 버블을
    // 그리지 않고 pending 버블이 대신 서 있다가, 첫 델타가 오면 여기가 자라기 시작한다.
    const placeholder: ChatMessage = {
      id: makeId("message"),
      role: "assistant",
      content: "",
      streaming: true,
    };
    // 오직 롤백용 스냅샷이다. 화면 갱신은 아래부터 전부 함수형 갱신으로 하는데, 델타가
    // 도착하는 동안 이 배열은 이미 낡은 값이 되므로 성공 경로에서 쓰면 델타가 통째로 날아간다.
    const previousMessages = messages;
    setMessages([...messages, optimisticUser, placeholder]);
    setQuestion("");
    setSending(true);
    setError("");
    setProgress("");

    try {
      let imageId: string | undefined;
      if (pendingFile) imageId = await uploadAsset(pendingFile);

      const result = await streamTurn(
        activeSessionId,
        {
          request_id: makeId("turn"),
          question: trimmed,
          ...(imageId ? { image: { image_id: imageId } } : {}),
          ...(clearImage ? { clear_image: true } : {}),
        },
        setProgress,
        (text) => {
          // 누적은 여기서 한다. id로 placeholder를 찾아 이어붙이므로, 그사이 사용자가 세션을
          // 지우거나 목록이 갈아치워졌다면 대상이 없어 조용히 아무 일도 일어나지 않는다.
          setMessages((prev) =>
            prev.map((message) =>
              message.id === placeholder.id
                ? { ...message, content: message.content + text }
                : message,
            ),
          );
        },
      );
      // done의 answer가 최종 진실이다. 누적한 델타를 신뢰하지 않고 통째로 교체하므로
      // 델타가 유실됐거나 툴 호출 태그가 섞여 나왔어도 턴이 끝나는 순간 교정된다.
      const assistant: ChatMessage = { ...placeholder, content: result.answer, streaming: false };
      setMessages((prev) =>
        result.history_reset
          ? [optimisticUser, assistant]
          : prev.map((message) => (message.id === placeholder.id ? assistant : message)),
      );
      const nextActiveImage = result.active_image
        ? {
            imageId: result.active_image.image_id,
            fileName: pendingFile?.name ?? activeImage?.fileName ?? "연결된 이미지",
            // 새로 고른 파일은 이미 브라우저가 성공적으로 그린 object URL을 그대로 넘긴다.
            // 턴 직후 서버 URL로 바꾸면 Agent가 재시작 전이거나 조회가 잠시 늦을 때 깨진다.
            previewUrl: pendingImageUrl || activeImage?.previewUrl || assetUrl(result.active_image.image_id),
            ownsPreviewUrl: Boolean(pendingImageUrl) || activeImage?.ownsPreviewUrl,
          }
        : null;
      if (pendingImageUrl && nextActiveImage) {
        pendingImageUrlRef.current = ""; // activeImage가 object URL 소유권을 넘겨받는다
      } else if (pendingImageUrl) {
        URL.revokeObjectURL(pendingImageUrl);
        pendingImageUrlRef.current = "";
      }
      replaceActiveImage(nextActiveImage);
      setPendingFile(null);
      setPendingImageUrl("");
      setClearImage(false);

      const now = Date.now();
      const next = sessions
        .map((session) =>
          session.id === activeSessionId
            ? {
                ...session,
                title: session.title === "새 대화" ? titleFrom(trimmed) : session.title,
                updatedAt: now,
              }
            : session,
        )
        .sort((a, b) => b.updatedAt - a.updatedAt);
      persistSessions(next);
    } catch (reason) {
      // 낙관적으로 넣은 user 메시지와 델타를 받던 placeholder를 스냅샷으로 함께 걷어낸다.
      setMessages(previousMessages);
      setQuestion(trimmed);
      setError(friendlyError(reason));
    } finally {
      setSending(false);
      setProgress("");
    }
  }

  function onComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  }

  async function removeSession(sessionId: string) {
    try {
      await deleteRemoteSession(sessionId);
    } catch {
      // Local navigation must remain usable while the on-device backend is offline.
    }
    const next = sessions.filter((session) => session.id !== sessionId);
    if (next.length) {
      persistSessions(next);
      if (sessionId === activeSessionId) setActiveSessionId(next[0].id);
    } else {
      const replacement = newSession();
      persistSessions([replacement]);
      setActiveSessionId(replacement.id);
      setMessages([]);
      replaceActiveImage(null);
    }
  }

  const statusLabel = backendState === "ready" ? "KAVA 준비됨" : backendState === "checking" ? "연결 확인 중" : "백엔드 연결 안 됨";

  /**
   * 첫 델타가 도착하기 전까지만 pending 버블을 세운다. 델타가 들어와 텍스트가 자라기
   * 시작하면 그 버블 자체가 진행 상황을 보여 주므로, 둘을 같이 띄우면 KAVA가 두 번
   * 답하는 것처럼 보인다. 진행 단계 문구(progress)도 pending 버블 안에만 둔다. 답변
   * 본문이 이미 흐르고 있는데 "답변을 작성하는 중"을 덧붙이는 것은 정보가 아니라 소음이다.
   */
  const isBlankPlaceholder = (message: ChatMessage) => Boolean(message.streaming) && !message.content;
  const showPending = sending && !messages.some((message) => message.streaming && message.content.length > 0);
  const previewImage = pendingFile && pendingImageUrl
    ? {
        url: pendingImageUrl,
        fileName: pendingFile.name,
        status: `${Math.ceil(pendingFile.size / 1024).toLocaleString()} KB · 전송 대기`,
        pending: true,
      }
    : activeImage
      ? {
          url: activeImage.previewUrl,
          fileName: activeImage.fileName,
          status: clearImage ? "다음 턴부터 이미지 문맥 해제" : "KLaVA가 보는 현재 이미지",
          pending: false,
        }
      : null;

  return (
    <main className="app-shell">
      <button
        className="mobile-menu"
        type="button"
        aria-label="대화 목록 열기"
        onClick={() => setSidebarOpen(true)}
      >
        ☰
      </button>

      {sidebarOpen && <button className="sidebar-scrim" aria-label="대화 목록 닫기" onClick={() => setSidebarOpen(false)} />}

      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`} aria-label="대화 목록">
        <div className="brand-row">
          <div className="brand-mark" aria-hidden="true">K</div>
          <div>
            <strong>KAVA</strong>
            <span>Local Vision Assistant</span>
          </div>
          <button className="close-sidebar" type="button" aria-label="대화 목록 닫기" onClick={() => setSidebarOpen(false)}>×</button>
        </div>

        <button className="new-chat" type="button" onClick={addSession}>
          <span aria-hidden="true">＋</span> 새 대화
        </button>

        <nav className="session-list" aria-label="저장된 대화">
          <p className="section-label">최근 대화</p>
          {sessions.map((session) => (
            <div className={`session-row ${session.id === activeSessionId ? "active" : ""}`} key={session.id}>
              <button type="button" onClick={() => { setActiveSessionId(session.id); setSidebarOpen(false); }}>
                <span className="session-icon" aria-hidden="true">◇</span>
                <span>{session.title}</span>
              </button>
              <button className="delete-chat" type="button" aria-label={`${session.title} 삭제`} onClick={() => void removeSession(session.id)}>×</button>
            </div>
          ))}
        </nav>

      </aside>

      <section className="chat-panel">
        <header className="chat-header">
          <div>
            <strong>{activeSession?.title ?? "새 대화"}</strong>
            <span>{activeImage ? "이미지 문맥 사용 중" : "텍스트 대화"}</span>
          </div>
          <div className={`connection-pill ${backendState}`}>
            <span aria-hidden="true" /> {statusLabel}
          </div>
        </header>

        {previewImage && (
          <aside className="image-dock" aria-label="현재 이미지">
            <button
              className={`image-tab ${imagePanelOpen ? "open" : ""}`}
              type="button"
              aria-label={imagePanelOpen ? "이미지 미리보기 닫기" : "이미지 미리보기 열기"}
              aria-expanded={imagePanelOpen}
              aria-controls="current-image-panel"
              onClick={() => setImagePanelOpen((open) => !open)}
            >
              <span className="image-tab-thumb" aria-hidden="true">
                {/* 로컬 object URL과 동적인 Agent 주소는 Next Image 최적화 경로를 거치지 않는다. */}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  key={`tab-${previewImage.url}`}
                  src={previewImage.url}
                  alt=""
                  onError={(event) => { event.currentTarget.hidden = true; }}
                />
                <span>▧</span>
              </span>
              <span>이미지</span>
              <span aria-hidden="true">{imagePanelOpen ? "▴" : "▾"}</span>
            </button>

            {imagePanelOpen && (
              <section className={`image-panel ${clearImage && !previewImage.pending ? "muted" : ""}`} id="current-image-panel">
                <div className="image-panel-frame">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    key={previewImage.url}
                    src={previewImage.url}
                    alt={`${previewImage.fileName} 미리보기`}
                    onError={(event) => { event.currentTarget.hidden = true; }}
                  />
                  <div className="image-preview-fallback"><span aria-hidden="true">▧</span>미리보기를 불러올 수 없습니다</div>
                </div>
                <div className="image-panel-meta">
                  <div><strong>{previewImage.fileName}</strong><span>{previewImage.status}</span></div>
                  {previewImage.pending ? (
                    <button type="button" onClick={clearPendingImage}>취소</button>
                  ) : (
                    <button type="button" onClick={() => setClearImage(!clearImage)}>{clearImage ? "유지" : "해제"}</button>
                  )}
                </div>
              </section>
            )}
          </aside>
        )}

        <div className="messages" aria-live="polite" ref={messagesRef}>
          {messages.length === 0 ? (
            <section className="empty-state">
              <div className="hero-mark" aria-hidden="true">K</div>
              <p className="eyebrow">PRIVATE · LOCAL · KOREAN</p>
              <h1>무엇을 함께 살펴볼까요?</h1>
              <p>질문을 입력하거나 이미지를 첨부하세요.</p>
              <div className="suggestions">
                {["이 문서의 핵심을 요약해줘", "표에서 가장 큰 값을 찾아줘", "이미지의 장면을 설명해줘"].map((suggestion) => (
                  <button type="button" key={suggestion} onClick={() => setQuestion(suggestion)}>{suggestion}<span aria-hidden="true">↗</span></button>
                ))}
              </div>
            </section>
          ) : (
            <div className="message-column">
              {messages.map((message) =>
                // 아직 한 글자도 못 받은 placeholder는 건너뛴다. 빈 버블 대신 pending 버블이 선다.
                isBlankPlaceholder(message) ? null : (
                  <article className={`message ${message.role}${message.streaming ? " streaming" : ""}`} key={message.id}>
                    <div className="avatar" aria-hidden="true">{message.role === "assistant" ? "K" : "나"}</div>
                    <div className="message-body">
                      <strong>{message.role === "assistant" ? "KAVA" : "사용자"}</strong>
                      {message.imageName && <div className="inline-image-chip">▧ {message.imageName}</div>}
                      {message.role === "assistant" ? (
                        // 스트리밍 중에도 그대로 마크다운으로 렌더한다. 닫히지 않은 코드펜스나
                        // 미완성 표 같은 부분 마크다운이 들어와도 remark가 그 시점의 최선으로
                        // 해석하고, 다음 델타에서 전체를 다시 파싱하며 자연히 교정된다.
                        <Markdown content={message.content} />
                      ) : (
                        // 사용자 메시지는 계속 평문이다. `#`으로 시작하는 질문이 제목으로 변해
                        // 버리는 것도 이상하고, 사용자 입력 역시 신뢰 경계 밖이라 마크다운
                        // 파서까지 태워 렌더 표면을 넓힐 이유가 없다.
                        <p>{message.content}</p>
                      )}
                    </div>
                  </article>
                ),
              )}
              {showPending && (
                <article className="message assistant pending" aria-label="KAVA가 답변을 생성하고 있습니다">
                  <div className="avatar" aria-hidden="true">K</div>
                  <div className="message-body">
                    <strong>KAVA</strong>
                    <div className="pending-row">
                      <div className="thinking"><i /><i /><i /></div>
                      {/* key를 문구로 두어 단계가 바뀔 때마다 페이드 인이 다시 돈다 */}
                      {progress && <span className="progress-label" key={progress}>{progress}</span>}
                    </div>
                  </div>
                </article>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <footer className="composer-wrap">
          {error && <div className="error-banner" role="alert"><span>!</span>{error}</div>}
          {pendingFile && messages.length > 0 && <p className="attachment-warning">새 이미지를 보내면 이전 대화 문맥이 초기화됩니다.</p>}

          <div className="composer">
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={onComposerKeyDown}
              placeholder={backendState === "offline" ? "백엔드를 실행한 뒤 질문하세요" : "KAVA에게 메시지 보내기"}
              aria-label="메시지 입력"
              rows={1}
              disabled={sending}
            />
            <div className="composer-actions">
              <div>
                <button className="icon-button" type="button" aria-label="이미지 첨부" onClick={() => fileInputRef.current?.click()} disabled={sending}>＋</button>
                <input ref={fileInputRef} className="visually-hidden" type="file" accept="image/*" onChange={selectFile} />
                <span>이미지</span>
              </div>
              <button className="send-button" type="button" aria-label="메시지 보내기" onClick={() => void sendMessage()} disabled={!question.trim() || sending}>↑</button>
            </div>
          </div>
          <p className="local-note">KAVA는 로컬 모델의 출력을 표시합니다. 중요한 정보는 원본과 대조하세요.</p>
        </footer>
      </section>
    </main>
  );
}
