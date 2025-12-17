"use client";

import { useState } from "react";
import { useToast } from "../../hooks/use-toast"; // ShadCN toast
import { Button } from "../ui/button";
import { ScrollArea } from "../ui/scroll-area";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "../ui/collapsible";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogOverlay,
} from "../ui/dialog";
import { Input } from "../ui/input";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "../ui/dropdown-menu";
import { ChevronRight, ChevronLeft, MoreHorizontal } from "lucide-react";
import { api } from "../../trpc/react";
import { useRouter, usePathname } from "next/navigation";
import { cn } from "../../lib/utils";
import { ThemeToggle } from "../theme-toggle";

interface Project {
  id: string;
  name: string;
}

export default function LeftSidebar() {
  const { toast } = useToast();
  const [isExpanded, setIsExpanded] = useState<boolean>(true);
  const [isAddDialogOpen, setIsAddDialogOpen] = useState<boolean>(false);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState<boolean>(false);
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState<boolean>(false);
  const [newProjectName, setNewProjectName] = useState<string>("");
  const [currentProject, setCurrentProject] = useState<Project | null>(null);

  const { data: projects = [], refetch } =
    api.project.getAllProjects.useQuery();
  const router = useRouter();
  const pathname = usePathname();

  const createProject = api.project.createProject.useMutation({
    onSuccess: (newProject) => {
      refetch();
      toast({ title: "Success", description: "Project created successfully." });
      router.push(`/project/${newProject.id}/dashboard`);
    },
    onError: (error) => {
      if (error.message.includes("already exists")) {
        toast({
          variant: "destructive",
          title: "Error",
          description: "Project name already exists.",
        });
      } else {
        toast({
          variant: "destructive",
          title: "Error",
          description: "Failed to create project.",
        });
      }
    },
  });

  const editProjectMutation = api.project.editProject.useMutation({
    onSuccess: () => {
      refetch();
      toast({ title: "Success", description: "Project updated successfully." });
    },
    onError: (error) => {
      if (error.message.includes("already exists")) {
        toast({
          variant: "destructive",
          title: "Error",
          description: "Project name already exists.",
        });
      } else {
        toast({
          variant: "destructive",
          title: "Error",
          description: "Failed to update project.",
        });
      }
    },
  });

  const deleteProjectMutation = api.project.deleteProject.useMutation({
    onSuccess: async (result) => {
      await refetch();
      if (result.createdNewUntitled && 'id' in result) {
        toast({ title: "Success", description: "Project deleted. New Untitled Project created." });
        router.push(`/project/${result.id}/dashboard`);
      } else {
        toast({ title: "Success", description: "Project deleted successfully." });
        const updatedProjects = await refetch();
        if (updatedProjects.data && updatedProjects.data.length > 0) {
          router.push(`/project/${updatedProjects.data[0]!.id}/dashboard`);
        }
      }
    },
    onError: () => {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to delete project.",
      });
    },
  });

  const addNewProject = async () => {
    if (newProjectName.trim()) {
      await createProject.mutateAsync({ name: newProjectName });
      setNewProjectName("");
      setIsAddDialogOpen(false);
    } else {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Project name cannot be empty.",
      });
    }
  };

  const editProject = async () => {
    if (currentProject && newProjectName.trim()) {
      await editProjectMutation.mutateAsync({
        projectId: currentProject.id,
        newName: newProjectName,
      });
      setNewProjectName("");
      setIsEditDialogOpen(false);
    } else {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Project name cannot be empty.",
      });
    }
  };

  const deleteProject = async () => {
    if (currentProject) {
      await deleteProjectMutation.mutateAsync({ projectId: currentProject.id });
      setCurrentProject(null);
      setIsDeleteDialogOpen(false);
    }
  };

  const handleProjectClick = (projectId: string) => {
    router.push(`/project/${projectId}/dashboard`);
  };

  const currentProjectId = pathname ? (pathname.match(/^\/project\/([^/]+)/)?.[1] ?? null) : null;

  return (
    <Collapsible
      open={isExpanded}
      onOpenChange={setIsExpanded}
      className="border-r bg-background"
    >
      <div className="flex h-[52px] items-center justify-between px-2">
        {isExpanded && <h2 className="text-lg font-semibold">Projects</h2>}
        <div className="flex items-center gap-2">
          {isExpanded && <ThemeToggle />}
          <CollapsibleTrigger asChild>
            <Button variant="ghost" size="icon" className="h-9 w-9">
              {isExpanded ? (
                <ChevronLeft className="h-4 w-4" />
              ) : (
                <ChevronRight className="h-4 w-4" />
              )}
            </Button>
          </CollapsibleTrigger>
        </div>
      </div>
      <CollapsibleContent className="space-y-2">
        <ScrollArea
          className={`h-[calc(100vh-52px)] ${isExpanded ? "w-64" : "w-0"}`}
        >
          <div className="flex flex-col gap-2 p-2">
            <Button
              variant="outline"
              className="flex items-center justify-between px-4"
              onClick={() => setIsAddDialogOpen(true)}
            >
              <span className="flex-grow text-center">New Project</span>
            </Button>

            {projects.map((project) => {
              const isActive = currentProjectId === project.id;
              return (
                <div
                  key={project.id}
                  className="flex items-center justify-between"
                >
                  <Button
                    variant={isActive ? "secondary" : "ghost"}
                    className={cn(
                      "justify-start p-3",
                      isActive && "bg-primary text-primary-foreground font-semibold hover:bg-primary hover:text-primary-foreground",
                      !isActive && "hover:bg-accent hover:text-accent-foreground"
                    )}
                    onClick={() => handleProjectClick(project.id)}
                  >
                    {project.name}
                  </Button>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="icon">
                        <MoreHorizontal className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent>
                      <DropdownMenuItem
                        onClick={() => {
                          setCurrentProject(project);
                          setNewProjectName(project.name);
                          setIsEditDialogOpen(true);
                        }}
                      >
                        Edit
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={() => {
                          setCurrentProject(project);
                          setIsDeleteDialogOpen(true);
                        }}
                      >
                        Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              );
            })}
          </div>
        </ScrollArea>
      </CollapsibleContent>

      <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
        <DialogOverlay />
        <DialogContent>
          <DialogHeader>
            <DialogTitle>New Project</DialogTitle>
          </DialogHeader>
          <Input
            type="text"
            placeholder="Enter project name"
            value={newProjectName}
            onChange={(e) => setNewProjectName(e.target.value)}
            className="mb-4"
          />
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsAddDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={addNewProject}>Add Project</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={isEditDialogOpen} onOpenChange={setIsEditDialogOpen}>
        <DialogOverlay />
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Project</DialogTitle>
          </DialogHeader>
          <Input
            type="text"
            placeholder="Enter new project name"
            value={newProjectName}
            onChange={(e) => setNewProjectName(e.target.value)}
            className="mb-4"
          />
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setIsEditDialogOpen(false)}
            >
              Cancel
            </Button>
            <Button onClick={editProject}>Save Changes</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
        <DialogOverlay />
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Confirm Delete</DialogTitle>
          </DialogHeader>
          <p>Are you sure you want to delete this project?</p>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setIsDeleteDialogOpen(false)}
            >
              Cancel
            </Button>
            <Button variant="destructive" onClick={deleteProject}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Collapsible>
  );
}