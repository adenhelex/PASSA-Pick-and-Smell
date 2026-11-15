import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PASSA pick and smell",
  description: "Local control panel for the MUI pick-and-smell arm.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
