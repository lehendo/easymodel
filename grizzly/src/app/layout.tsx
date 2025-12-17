import "@/styles/globals.css";

import { GeistSans } from "geist/font/sans";
import { type Metadata } from "next";

import { TRPCReactProvider } from "../trpc/react";

import { ThemeProvider } from "../components/theme-provider";

import { Toaster } from "../components/ui/toaster";

// Ensure the root layout is treated as dynamic to avoid build-time data fetching
export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "EasyModel Studio",
  description: "No-code model fine-tuning with React Flow",
  icons: [{ rel: "icon", url: "/favicon.ico" }],
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${GeistSans.variable}`}
      suppressHydrationWarning
    >
      <body>
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <TRPCReactProvider>{children}</TRPCReactProvider>
          <Toaster />
        </ThemeProvider>
      </body>
    </html>
  );
}
