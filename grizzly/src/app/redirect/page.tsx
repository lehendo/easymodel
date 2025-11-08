"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import { api } from "@/trpc/react";

export default function RedirectPage() {
  const router = useRouter();

  // Fetch all projects for the current user
  const getAllProjects = api.project.getAllProjects.useQuery(undefined, {
    refetchOnWindowFocus: false, // Avoid unnecessary re-fetching
    staleTime: 60 * 1000, // Cache for 1 minute
  });

  useEffect(() => {
    if (getAllProjects.isFetched) {
      const projects = getAllProjects.data || [];

      if (projects.length > 0) {
        // Preload the data for each project's dashboard and analytics pages
        projects.forEach((project) => {
          // Prefetch the dashboard and analytics pages for this project
          router.prefetch(`/project/${project.id}/dashboard`);
          router.prefetch(`/project/${project.id}/analytics`);
        });

        // Wait for 10 seconds before redirecting
        setTimeout(() => {
          router.replace(`/project/${projects[0].id}/dashboard`);
        }, 3000); // Delay for 10 seconds
      } else {
        console.error("No projects found, but a default project should exist.");
      }
    }
  }, [getAllProjects.isFetched, getAllProjects.data, router]);

  return (
    <div className="flex h-screen items-center justify-center">
      <img
        src="favicon.svg" // Use the favicon path
        alt="Loading..."
        width={64} // Set the size of the favicon
        height={64} // Set the size of the favicon
        className="animate-fade-grow" // Apply the glow animation
      />
    </div>
  );
}
