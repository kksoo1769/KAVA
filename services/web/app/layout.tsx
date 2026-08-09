import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "KAVA — 온디바이스 비전 어시스턴트",
  description: "한국어 텍스트와 이미지를 로컬에서 처리하는 온디바이스 KAVA 채팅 인터페이스",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
