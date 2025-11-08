"use client";

import { memo } from "react";
import { Dock, DockIcon } from "../ui/dock";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../ui/tooltip";
import { HomeIcon, ChartNoAxesColumn } from "lucide-react";
import Link from "next/link";
import { cn } from "../../lib/utils";
import { buttonVariants } from "../ui/button";
import { usePathname } from "next/navigation";

const DockComponent = memo(() => {
  const pathname = usePathname(); // Get the current pathname

  // Extract projectId from URL using regex
  const projectId = (() => {
    const regex = /^\/project\/([^/]+)(\/(dashboard|analytics))?/; // Matches '/projects/{projectId}/dashboard' or '/projects/{projectId}/analytics'
    const match = pathname?.match(regex);
    return match ? match[1] : null; // Return the projectId if found
  })();

  // If no projectId is found, return null to avoid rendering
  if (!projectId) return null;

  const DATA = [
    {
      href: `/project/${projectId}/dashboard`, // Use projectId in the link
      icon: HomeIcon,
      label: "Dashboard",
    },
    {
      href: `/project/${projectId}/analytics`, // Use projectId in the link
      icon: ChartNoAxesColumn,
      label: "Analytics",
    },
  ];

  return (
    <div className="absolute -top-1 left-1/2 z-10 -translate-x-1/2 transform">
      <TooltipProvider>
        <Dock>
          {DATA.map((item) => {
            const isActive = pathname === item.href; // Check if the current path matches the item href

            return (
              <DockIcon key={item.label}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Link
                      href={item.href}
                      prefetch={true}
                      aria-label={item.label}
                      className={cn(
                        buttonVariants({ variant: "ghost", size: "icon" }),
                        "size-12 rounded-full",
                        isActive
                          ? "bg-primary p-3 text-primary-foreground" // Highlight active item
                          : "",
                      )}
                    >
                      <item.icon className="size-4" />
                    </Link>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>{item.label}</p>
                  </TooltipContent>
                </Tooltip>
              </DockIcon>
            );
          })}
        </Dock>
      </TooltipProvider>
    </div>
  );
});

DockComponent.displayName = "DockComponent"; // Display name for debugging purposes

export default DockComponent;
