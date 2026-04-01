import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Chest X-ray Analysis",
  description: "FastAPI and Next.js demo for chest X-ray analysis, Grad-CAM, and medical report generation.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
