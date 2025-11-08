"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/trpc/react";

export default function Home() {
  const router = useRouter();
  const { data: projects = [], isLoading } = api.project.getAllProjects.useQuery();

  useEffect(() => {
    if (!isLoading && projects.length > 0 && projects[0]) {
      // Redirect to first project's dashboard
      router.replace(`/project/${projects[0].id}/dashboard`);
    }
  }, [isLoading, projects, router]);

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-lg">Loading...</div>
      </div>
    );
  }

  // If no projects, show message (shouldn't happen as default project is created)
  if (projects.length === 0) {
    return (
      <main className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-b from-[#2e026d] to-[#15162c] text-white">
        <div className="container flex flex-col items-center justify-center gap-12 px-4 py-16">
          <h1 className="text-5xl font-extrabold tracking-tight sm:text-[5rem]">
            Welcome to <span className="text-[hsl(280,100%,70%)]">EasyModel</span>
          </h1>
          <p className="text-xl">Setting up your workspace...</p>
        </div>
      </main>
    );
  }

  // Loading state while redirecting
  return (
    <div className="flex min-h-screen items-center justify-center">
      <div className="text-lg">Redirecting to dashboard...</div>
    </div>
  );
}
