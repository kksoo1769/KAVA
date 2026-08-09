import { createServer } from "node:http";

const sessions = new Map();
const requests = new Map();
let assetSequence = 0;
const previewPng = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Z4h8AAAAASUVORK5CYII=",
  "base64",
);

const CORS = {
  "access-control-allow-origin": "http://127.0.0.1:3100",
  "access-control-allow-methods": "GET,POST,DELETE,OPTIONS",
  "access-control-allow-headers": "content-type",
};

function json(response, status, body) {
  response.writeHead(status, { "content-type": "application/json; charset=utf-8", ...CORS });
  response.end(JSON.stringify(body));
}

/** 실제 8000이 내는 것과 같은 모양의 SSE 프레임. event 줄, data 줄, 빈 줄. */
function sse(event, data) {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

/**
 * 개행을 CRLF로 쓰는 SSE 프레임. 실제 백엔드는 LF를 쓰지만 SSE 스펙은 CRLF도 허용하고,
 * 중간에 끼는 프록시나 다른 언어의 서버 구현은 CRLF로 내보낸다. 프레임 구분자를 `\n\n`
 * 리터럴로 찾으면 CRLF 스트림에서는 프레임이 하나도 잘리지 않아 답변이 통째로 사라진다.
 */
function sseCrlf(event, data) {
  return `event: ${event}\r\ndata: ${JSON.stringify(data)}\r\n\r\n`;
}

async function body(request) {
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  return Buffer.concat(chunks);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * 답변을 델타 조각으로 쪼갠다. 줄 단위로 자르고 개행은 앞 조각 끝에 붙이므로, 들여쓴
 * 줄은 조각이 공백으로 시작한다. 프론트엔드가 선행 공백을 보존하지 못하면 중첩 리스트가
 * 평평해지므로 렌더 결과로 바로 드러난다. 긴 줄은 한 번 더 반으로 갈라, 문장이 여러
 * 델타에 걸쳐 도착해도 호출부의 누적이 제대로 이어지는지 함께 태운다.
 */
function deltaChunks(answer) {
  const lines = answer.split("\n");
  const chunks = [];
  lines.forEach((line, index) => {
    const tail = index === lines.length - 1 ? "" : "\n";
    if (line.length > 28) {
      const cut = Math.ceil(line.length / 2);
      chunks.push(line.slice(0, cut), line.slice(cut) + tail);
    } else {
      chunks.push(line + tail);
    }
  });
  return chunks.filter((chunk) => chunk.length > 0);
}

/**
 * 마크다운 렌더링을 검증하기 위한 덧붙임 블록. 기존 e2e가 의존하는 한 줄 답변 패턴
 * (`로컬 모델 답변: …`, `이미지 문맥에서 답변했습니다: …`)을 건드리지 않으려고 특정 질문에만
 * 반응하는 분기로 둔다. GFM 표, 제목, 중첩 리스트, 인라인 코드, 인용을 모두 담는다.
 * `비고` 열은 일부러 길게 두어 표가 버블 폭을 넘기고 래퍼가 가로 스크롤을 갖게 만든다.
 */
const MARKDOWN_TRIGGER = "마크다운";

/**
 * 얌전하지 않은 서버를 흉내내는 분기. CRLF로 프레임을 끊고, 마지막 done 프레임 뒤에는
 * 빈 줄을 붙이지 않은 채 연결을 닫는다. 둘 다 실제로 있을 수 있는 변형인데, 프론트엔드가
 * 프레임 구분자를 `\n\n`으로만 찾거나 스트림 종료 후 남은 버퍼를 버리면 정상 응답을
 * 받았는데도 "답변을 받지 못한 채 연결이 끊겼습니다"가 뜬다.
 */
const ROUGH_TRIGGER = "거친 스트림";
/*
 * 중첩 리스트를 표보다 앞에 둔다. 델타로 누적된 중간 상태는 done이 오는 순간 원문으로
 * 교체되므로, "선행 공백이 전송 구간에서 살아남았는가"는 스트리밍이 끝나기 전에만 관찰할
 * 수 있다. 들여쓴 줄을 앞쪽에 두면 그 뒤로 표의 긴 줄들이 흐르는 동안 관찰 창이 넉넉하게
 * 열려 있어서, 타이밍에 기대는 어서션이 안정적으로 붙는다.
 */
const MARKDOWN_BLOCK = `### 온디바이스 모델 비교

주의할 점:

- 메모리는 4bit 양자화 기준이다
  - 8bit로 올리면 약 1.8배가 된다
- OCR 정확도는 \`--ocr-merge\` 옵션에 따라 달라진다

| 모델 | 파라미터 | 메모리 | 한국어 | 비고 |
| --- | ---: | ---: | --- | --- |
| KAVA-mini | 2.4B | 3.1 GB | 좋음 | 4bit 양자화 기준이며 OCR 병합 옵션을 켠 상태에서 같은 문서 열 장을 반복 측정한 값이다 |
| KAVA-base | 7.8B | 9.6 GB | 아주 좋음 | 표 렌더링과 장문 요약에서 가장 안정적이고 첫 토큰까지의 지연도 준수한 편이라 기본값으로 쓴다 |
| KAVA-vision | 11.2B | 14.2 GB | 좋음 | 이미지 문맥을 함께 물릴 때 권장하는 구성이며 메모리 여유가 넉넉한 기기에서만 골라야 한다 |

> 위 수치는 문서용 예시 값이다.`;

/** 한 턴의 상태 변화와 응답 본문. JSON 경로와 SSE 경로가 이 하나를 함께 쓴다. */
function runTurn(sessionId, payload) {
  const dedupeKey = `${sessionId}:${payload.request_id}`;
  if (requests.has(dedupeKey)) return requests.get(dedupeKey);

  let state = sessions.get(sessionId) ?? { messages: [], active_image: null };
  let historyReset = false;
  if (payload.image && payload.image.image_id !== state.active_image?.image_id) {
    state = { messages: [], active_image: { image_id: payload.image.image_id, kind: "photo" } };
    historyReset = true;
  } else if (payload.clear_image) {
    state = { messages: [], active_image: null };
    historyReset = true;
  }
  const headline = state.active_image
    ? `이미지 문맥에서 답변했습니다: ${payload.question}`
    : `로컬 모델 답변: ${payload.question}`;
  const answer = payload.question.includes(MARKDOWN_TRIGGER) ? `${headline}\n\n${MARKDOWN_BLOCK}` : headline;
  state.messages.push({ role: "user", content: payload.question }, { role: "assistant", content: answer });
  sessions.set(sessionId, state);
  const result = {
    session_id: sessionId,
    answer,
    turn_count: state.messages.length / 2,
    history_reset: historyReset,
    active_image: state.active_image,
  };
  requests.set(dedupeKey, result);
  return result;
}

const server = createServer(async (request, response) => {
  if (request.method === "OPTIONS") return json(response, 204, {});
  const url = new URL(request.url ?? "/", "http://127.0.0.1:8100");
  if (url.pathname === "/readyz") return json(response, 200, { status: "ready" });
  if (request.method === "POST" && url.pathname === "/v1/assets") {
    await body(request);
    assetSequence += 1;
    return json(response, 201, { image_id: `asset-${assetSequence}` });
  }
  if (request.method === "GET" && /^\/v1\/assets\/[^/]+$/.test(url.pathname)) {
    response.writeHead(200, {
      "content-type": "image/png",
      "cache-control": "private, max-age=3600",
      ...CORS,
    });
    return response.end(previewPng);
  }

  const turnMatch = url.pathname.match(/^\/v1\/sessions\/([^/]+)\/turns$/);
  const streamMatch = url.pathname.match(/^\/v1\/sessions\/([^/]+)\/turns\/stream$/);
  const sessionMatch = url.pathname.match(/^\/v1\/sessions\/([^/]+)$/);
  if (request.method === "POST" && turnMatch) {
    const payload = JSON.parse((await body(request)).toString("utf8"));
    return json(response, 200, runTurn(decodeURIComponent(turnMatch[1]), payload));
  }
  if (request.method === "POST" && streamMatch) {
    const payload = JSON.parse((await body(request)).toString("utf8"));
    const result = runTurn(decodeURIComponent(streamMatch[1]), payload);
    response.writeHead(200, {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache",
      ...CORS,
    });
    const rough = payload.question.includes(ROUGH_TRIGGER);
    const encode = rough ? sseCrlf : sse;

    // 프레임을 따로따로 흘려보낸다. 한 덩어리로 보내면 프론트엔드의 버퍼 이어붙이기
    // 경로가 한 번도 실행되지 않아 e2e가 통과해도 실제 동작을 보증하지 못한다.
    for (const label of ["이미지 종류를 확인하는 중", "답변을 작성하는 중"]) {
      response.write(encode("progress", { label }));
      await sleep(90);
    }

    // 이어서 답변을 토큰 델타처럼 조각내 흘려보낸다. 백엔드가 아직 delta를 보내지 않으므로
    // 프론트엔드의 수신 구조를 검증할 수 있는 곳은 지금 이 목뿐이다.
    for (const [index, text] of deltaChunks(result.answer).entries()) {
      const frame = encode("delta", { text });
      if (index === 0) {
        // 첫 프레임은 일부러 두 번의 write로 쪼갠다. chunk 경계가 SSE 프레임 중간에
        // 떨어지는 경우인데, streamTurn의 "미완 조각을 버퍼에 되돌려 다음 chunk와 이어
        // 붙이기" 경로는 오직 이때만 실행된다. 지금까지 어느 테스트도 이 경로를 태우지
        // 않아서, 그 로직이 깨져도 e2e는 초록으로 통과했다.
        // 자르는 위치는 `data: {"t` 바로 뒤로 잡는다. ASCII 구간이라 두 조각을 각각
        // UTF-8로 인코딩해도 멀티바이트 문자가 반토막 나지 않는다.
        const cut = frame.indexOf("data:") + 8;
        response.write(frame.slice(0, cut));
        await sleep(25);
        response.write(frame.slice(cut));
      } else {
        response.write(frame);
      }
      await sleep(55);
    }

    // 본문은 JSON 경로와 같은 것을 그대로 쓴다. rough 모드에서는 프레임을 닫는 빈 줄을
    // 빼고 그대로 연결을 끊어, 마지막 프레임이 버퍼에 갇히는 상황을 재현한다.
    response.write(rough ? encode("done", result).replace(/\r\n\r\n$/, "") : sse("done", result));
    return response.end();
  }
  if (request.method === "GET" && sessionMatch) {
    const sessionId = decodeURIComponent(sessionMatch[1]);
    const state = sessions.get(sessionId);
    if (!state) return json(response, 404, { code: "session_not_found", detail: "세션이 없습니다." });
    return json(response, 200, {
      session_id: sessionId,
      turn_count: state.messages.length / 2,
      messages: state.messages,
      active_image: state.active_image,
      has_ocr_cache: false,
    });
  }
  if (request.method === "DELETE" && sessionMatch) {
    const sessionId = decodeURIComponent(sessionMatch[1]);
    const deleted = sessions.delete(sessionId);
    return json(response, 200, { session_id: sessionId, deleted });
  }
  return json(response, 404, { detail: "not found" });
});

server.listen(8100, "127.0.0.1");
