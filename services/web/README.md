# KAVA On-device Web

온디바이스 KAVA를 `localhost`에서 사용하는 한국어 채팅 UI입니다. 브라우저에는 대화 목록의 표시 정보만 저장하고, 실제 메시지와 이미지 상태는 `services/agent/test.py`에서 리팩터링할 KAVA Agent FastAPI가 관리합니다.

## 실행

```bash
pnpm install
cp .env.example .env.local
pnpm dev
```

기본 주소는 다음과 같습니다.

- 웹 UI: `http://127.0.0.1:3000`
- KAVA Agent FastAPI: `http://127.0.0.1:8000`
- 내부 KLaVA FastAPI: `http://127.0.0.1:8001`

## 검증

```bash
pnpm build
node --test tests/rendered-html.test.mjs
pnpm exec playwright install chromium
pnpm test:e2e
```

E2E 테스트는 포트 `8100`에 계약용 가짜 Agent API를 자동으로 띄우므로 실제 8000·8001 서버를 변경하지 않습니다.

Web UI는 내부 KLaVA 8001을 직접 호출하지 않습니다. UI가 요구하는 `/readyz`, `/v1/assets`, `/v1/sessions/{session_id}`, `/v1/sessions/{session_id}/turns`는 KAVA Agent FastAPI 8000에 [온디바이스 웹 가이드](../../analysis/KAVA_ON_DEVICE_WEB_GUIDE.md)에 따라 구현합니다. Agent API가 KLaVA 8001과 OCR 8002를 내부 adapter로 호출합니다.

백엔드 구현 계약과 온디바이스 구성은 [`../../analysis/KAVA_ON_DEVICE_WEB_GUIDE.md`](../../analysis/KAVA_ON_DEVICE_WEB_GUIDE.md)를 참고하세요.
