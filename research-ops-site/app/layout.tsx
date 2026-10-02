import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Research OS · DeReFusion",
  description: "Evidence-first research operations dashboard for DeReFusion.",
  other: {
    "codex-preview": "development",
  },
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body className="antialiased">{children}</body>
    </html>
  );
}
