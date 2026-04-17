import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "polyglot-demo",
  description: "Open-source video translation demo (CPU-only).",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-ink-900 text-ink-200 antialiased">
        {children}
      </body>
    </html>
  );
}
