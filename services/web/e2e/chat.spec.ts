import { expect, test, type Page } from "@playwright/test";

/**
 * 완성된 답변 버블을 문구로 찾는다.
 *
 * `getByText`와 달리 `hasText`는 버블 전체의 텍스트를 보므로, 마크다운 렌더링이 문장을
 * 여러 DOM 노드로 쪼개도 놓치지 않는다. `:not(.streaming)`으로 좁히는 것도 중요하다.
 * 짧은 답변은 델타 한 조각으로 다 도착하기 때문에, 이 조건이 없으면 done이 오기 전
 * (즉 history_reset이 아직 적용되지 않은 시점)에 어서션이 통과해 뒤따르는 "이전 답변이
 * 사라졌다" 검사가 경합으로 깨진다.
 */
function answerBubble(page: Page, text: string) {
  return page.locator("article.message.assistant:not(.pending):not(.streaming)", { hasText: text });
}

test("텍스트 멀티 턴과 새 대화를 처리한다", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("KAVA 준비됨")).toBeVisible();

  const composer = page.getByLabel("메시지 입력");
  await composer.fill("내 이름은 민수야");
  await page.getByLabel("메시지 보내기").click();
  await expect(answerBubble(page, "로컬 모델 답변: 내 이름은 민수야")).toHaveCount(1);

  await composer.fill("내 이름을 기억해?");
  await page.getByLabel("메시지 보내기").click();
  await expect(answerBubble(page, "로컬 모델 답변: 내 이름을 기억해?")).toHaveCount(1);
  await expect(page.locator("article.message.user")).toHaveCount(2);
  // pending 버블도 article.message.assistant라서, 스트리밍 중에 세면 카운트가 어긋난다.
  await expect(page.locator("article.message.assistant:not(.pending)")).toHaveCount(2);

  await page.getByRole("button", { name: "새 대화" }).click();
  await expect(page.getByText("무엇을 함께 살펴볼까요?")).toBeVisible();
});

test("이미지 등록, 문맥 초기화, 이미지 해제를 처리한다", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("KAVA 준비됨")).toBeVisible();
  const composer = page.getByLabel("메시지 입력");
  await composer.fill("먼저 텍스트 질문");
  await page.getByLabel("메시지 보내기").click();
  await expect(answerBubble(page, "로컬 모델 답변: 먼저 텍스트 질문")).toHaveCount(1);

  await page.getByLabel("이미지 첨부").click();
  await page.locator('input[type="file"]').setInputFiles({
    name: "sample.png",
    mimeType: "image/png",
    buffer: Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Z4h8AAAAASUVORK5CYII=",
      "base64",
    ),
  });
  const imagePreview = page.getByRole("img", { name: "sample.png 미리보기" });
  await expect(imagePreview).toBeVisible();
  const localPreviewUrl = await imagePreview.getAttribute("src");
  expect(localPreviewUrl).toMatch(/^blob:/);
  await expect(page.getByText("새 이미지를 보내면 이전 대화 문맥이 초기화됩니다.")).toBeVisible();
  await composer.fill("이 이미지가 뭐야?");
  await page.getByLabel("메시지 보내기").click();
  await expect(answerBubble(page, "이미지 문맥에서 답변했습니다: 이 이미지가 뭐야?")).toHaveCount(1);
  await expect(answerBubble(page, "로컬 모델 답변: 먼저 텍스트 질문")).toHaveCount(0);
  await expect(imagePreview).toBeVisible();
  await expect(imagePreview).toHaveAttribute("src", localPreviewUrl!);
  await expect(page.getByText("KLaVA가 보는 현재 이미지")).toBeVisible();

  await page.getByRole("button", { name: "이미지 미리보기 닫기" }).click();
  await expect(imagePreview).toHaveCount(0);
  await page.getByRole("button", { name: "이미지 미리보기 열기" }).click();
  await expect(imagePreview).toBeVisible();

  await page.getByRole("button", { name: "해제" }).click();
  await composer.fill("이제 텍스트로 답해줘");
  await page.getByLabel("메시지 보내기").click();
  await expect(answerBubble(page, "로컬 모델 답변: 이제 텍스트로 답해줘")).toHaveCount(1);
  await expect(answerBubble(page, "이미지 문맥에서 답변했습니다: 이 이미지가 뭐야?")).toHaveCount(0);
});

test("마크다운 답변을 표, 제목, 리스트로 렌더한다", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("KAVA 준비됨")).toBeVisible();
  await page.getByLabel("메시지 입력").fill("마크다운으로 정리해줘");
  await page.getByLabel("메시지 보내기").click();

  const bubble = answerBubble(page, "로컬 모델 답변: 마크다운으로 정리해줘");
  await expect(bubble).toHaveCount(1);

  // 마크다운 문법이 태그로 바뀌었는지 본다.
  await expect(bubble.locator(".markdown h3")).toHaveText("온디바이스 모델 비교");
  await expect(bubble.locator(".markdown-table-scroll")).toHaveCount(1);
  await expect(bubble.locator("table thead th")).toHaveCount(5);
  await expect(bubble.locator("table thead th").first()).toHaveText("모델");
  await expect(bubble.locator("table tbody tr")).toHaveCount(3);
  await expect(bubble.locator(".markdown > ul > li")).toHaveCount(2);
  // 들여쓴 줄이 중첩 리스트로 살아남았다는 뜻이다. SSE 파서가 선행 공백을 깎아 버리면
  // 이 리스트가 평평해져서 카운트가 0이 된다.
  await expect(bubble.locator(".markdown ul ul > li")).toHaveCount(1);
  await expect(bubble.locator(".markdown blockquote")).toContainText("문서용 예시 값");
  await expect(bubble.locator(".markdown code")).toHaveText("--ocr-merge");

  // 원문 문법 문자가 화면 텍스트로 새지 않는다.
  const shown = await bubble.innerText();
  expect(shown).not.toContain("###");
  expect(shown).not.toContain("|---");
  expect(shown).not.toContain("| 모델 |");
  expect(shown).not.toContain("`--ocr-merge`");

  // 표는 래퍼 안에서만 넘친다. 레이아웃 전체가 옆으로 밀리면 안 된다.
  const tableScrolls = await bubble
    .locator(".markdown-table-scroll")
    .evaluate((element) => element.scrollWidth > element.clientWidth);
  expect(tableScrolls).toBe(true);
  const pageOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(pageOverflow).toBeLessThanOrEqual(1);
});

test("델타를 이어붙여 답변이 완성되기 전에 보여준다", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("KAVA 준비됨")).toBeVisible();
  await page.getByLabel("메시지 입력").fill("마크다운으로 스트리밍을 확인하자");
  await page.getByLabel("메시지 보내기").click();

  // 첫 델타가 오기 전에는 pending 버블이 진행 단계를 대신 보여준다. 목이 progress 두 개를
  // 90ms 간격으로 먼저 보내므로 이 구간이 200ms 가까이 열려 있다.
  await expect(page.locator("article.message.assistant.pending")).toBeVisible();

  // 델타가 도착하면 pending은 물러나고, 아직 자라는 중인 버블이 부분 텍스트를 들고 나온다.
  // 이 어서션이 통과하는 것 자체가 "답변이 완성되기 전에 화면에 보였다"는 증거다.
  // .streaming 클래스는 done 이벤트가 오는 순간 사라지기 때문이다.
  const streaming = page.locator("article.message.assistant.streaming");
  await expect(streaming).toContainText("로컬 모델 답변: 마크다운으로 스트리밍을 확인하자");
  await expect(page.locator("article.message.assistant.pending")).toHaveCount(0);

  // 아직 자라는 중인 텍스트만으로도 중첩 리스트가 만들어진다. 즉 들여쓰기 공백이 SSE
  // 전송 구간에서 살아남았다는 뜻이다. 이 검사는 done이 오기 전에만 의미가 있다. done의
  // answer가 최종 진실이라 스트리밍이 끝나면 어차피 온전한 원문으로 교체되기 때문이다.
  await expect(streaming.locator(".markdown ul ul > li")).toHaveCount(1);
  // 부분 마크다운도 렌더된다. 표가 아직 다 오지 않았어도 앞쪽 블록은 이미 태그가 되어 있다.
  await expect(streaming.locator(".markdown h3")).toHaveText("온디바이스 모델 비교");

  // done이 오면 표식이 내려가고 answer로 확정된다. 마지막 조각까지 도착했는지 함께 본다.
  await expect(streaming).toHaveCount(0);
  await expect(answerBubble(page, "위 수치는 문서용 예시 값이다")).toHaveCount(1);
});

test("CRLF 프레임과 빈 줄 없이 끝나는 스트림도 답변으로 받는다", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("KAVA 준비됨")).toBeVisible();
  // 목이 이 질문에만 CRLF로 프레임을 끊고, done 프레임 뒤 빈 줄을 빼고 연결을 닫는다.
  await page.getByLabel("메시지 입력").fill("거친 스트림으로 답해줘");
  await page.getByLabel("메시지 보내기").click();

  await expect(answerBubble(page, "로컬 모델 답변: 거친 스트림으로 답해줘")).toHaveCount(1);
  // done을 놓치면 이 배너가 뜬다. 스트림을 정상 종료로 인식했는지 함께 확인한다.
  await expect(page.getByRole("alert")).toHaveCount(0);
});

test("모바일에서 대화 목록을 열고 닫는다", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.getByText("KAVA 준비됨")).toBeVisible();
  await page.getByLabel("대화 목록 열기").click();
  await expect(page.getByRole("complementary", { name: "대화 목록" })).toHaveClass(/sidebar-open/);
  await page.getByLabel("대화 목록 닫기").first().click();
  await expect(page.getByRole("complementary", { name: "대화 목록" })).not.toHaveClass(/sidebar-open/);
});
