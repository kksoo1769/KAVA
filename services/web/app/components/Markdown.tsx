"use client";

import { memo } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * 답변 텍스트 안의 원시 HTML은 절대 살리지 않는다. rehype-raw 같은 플러그인을 넣으면
 * 답변에 섞인 `<script>`나 `<img onerror=...>`가 그대로 실행되는데, 답변은 로컬 LLM이
 * 생성하고 그 안에는 웹 검색 도구가 가져온 외부 데이터가 섞여 있다. 즉 신뢰 경계 밖의
 * 입력이므로 react-markdown의 기본값(원시 HTML을 텍스트로 무시)을 그대로 유지한다.
 * 링크 URL도 기본 urlTransform이 javascript: 같은 스킴을 걸러 준다.
 */
const components: Components = {
  // 채팅 버블은 좁아서 표가 조금만 넓어도 레이아웃 전체를 옆으로 밀어낸다. 표를 래퍼로
  // 감싸 가로 스크롤을 래퍼가 흡수하게 한다(스타일은 .markdown-table-scroll).
  table: ({ children }) => (
    <div className="markdown-table-scroll">
      <table>{children}</table>
    </div>
  ),
  // 외부 링크는 새 탭으로 열고, noopener로 원본 탭 탈취(window.opener) 경로를 막는다.
  a: ({ href, title, children }) => (
    <a href={href} title={title} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  ),
};

/**
 * 메시지 목록은 새 토큰이 도착할 때마다 전체가 리렌더된다. memo로 감싸 두면 content가
 * 그대로인 과거 메시지는 마크다운 파싱을 다시 하지 않는다. 파싱 비용은 글자 수에
 * 비례하므로, 대화가 길어질수록 이 memo가 스트리밍 프레임 드랍을 막아 준다.
 */
const Markdown = memo(function Markdown({ content }: { content: string }) {
  return (
    <div className="markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {content}
      </ReactMarkdown>
    </div>
  );
});

export default Markdown;
