import { z } from "zod";
import { createTRPCRouter, publicProcedure } from "@/server/api/trpc";
import { TRPCError } from "@trpc/server";

export const projectRouter = createTRPCRouter({
  // Fetch all projects for anonymous user
  getAllProjects: publicProcedure.query(async ({ ctx }) => {
    try {
      // Get or create default user
      let defaultUser = await ctx.db.user.findFirst({
        where: { clerkUID: "anonymous" },
      });

      if (!defaultUser) {
        defaultUser = await ctx.db.user.create({
          data: {
            clerkUID: "anonymous",
            email: "anonymous@easymodel.local",
          },
        });
      }

      // Return all projects for the anonymous user
      const projects = await ctx.db.project.findMany({
        where: { userId: defaultUser.id },
        orderBy: { createdAt: "desc" },
      });

      // If no projects exist, create an "Untitled Project"
      if (projects.length === 0) {
        const untitledProject = await ctx.db.project.create({
          data: {
            name: "Untitled Project",
            userId: defaultUser.id,
            nodes: JSON.stringify([]), // Empty array for blank canvas
            analytics: {
              create: {
                data: JSON.stringify({}),
              },
            },
          },
        });

        return [{
          ...untitledProject,
          nodes: [],
        }];
      }

      // Parse JSON string back to object for each project
      return projects.map(project => ({
        ...project,
        nodes: typeof project.nodes === 'string' ? JSON.parse(project.nodes) : project.nodes,
      }));
    } catch (error) {
      // Database not available - return in-memory default project
      console.warn("Database error, returning in-memory project:", error);
      return [{
        id: ctx.defaultProjectId || "untitled-project-in-memory",
        name: "Untitled Project",
        nodes: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      }];
    }
  }),

  // Fetch a specific project by its ID
  getProjectById: publicProcedure
    .input(z.object({ projectId: z.string() }))
    .query(async ({ ctx, input }) => {
      try {
        // Find the project by ID (no auth required)
        const project = await ctx.db.project.findFirst({
          where: {
            id: input.projectId,
          },
        });

        if (!project) {
          throw new TRPCError({
            code: "NOT_FOUND",
            message: "Project not found",
          });
        }

        // Parse JSON string to object
        return {
          ...project,
          nodes: typeof project.nodes === 'string' ? JSON.parse(project.nodes) : project.nodes,
        };
      } catch (error) {
        // If database error, return in-memory project
        if (input.projectId === ctx.defaultProjectId || input.projectId === "untitled-project-in-memory") {
          return {
            id: input.projectId,
            name: "Untitled Project",
            nodes: [],
            createdAt: new Date(),
            updatedAt: new Date(),
          };
        }
        throw error;
      }
    }),

  // Create a new project
  createProject: publicProcedure
    .input(z.object({ name: z.string().min(1) }))
    .mutation(async ({ ctx, input }) => {
      // Get or create default user
      let defaultUser = await ctx.db.user.findFirst({
        where: { clerkUID: "anonymous" },
      });

      if (!defaultUser) {
        defaultUser = await ctx.db.user.create({
          data: {
            clerkUID: "anonymous",
            email: "anonymous@easymodel.local",
          },
        });
      }

      const existingProject = await ctx.db.project.findFirst({
        where: {
          name: input.name,
          userId: defaultUser.id,
        },
      });

      if (existingProject) {
        throw new TRPCError({
          code: "BAD_REQUEST",
          message: "A project with the same name already exists.",
        });
      }

      const project = await ctx.db.project.create({
        data: {
          name: input.name,
          userId: defaultUser.id,
          nodes: JSON.stringify({}), // Store as JSON string for SQLite
          analytics: {
            create: {
              data: JSON.stringify({}), // Store as JSON string for SQLite
            },
          },
        },
      });

      // Parse JSON string back to object
      return {
        ...project,
        nodes: typeof project.nodes === 'string' ? JSON.parse(project.nodes) : project.nodes,
      };
    }),

  // Delete a project
  deleteProject: publicProcedure
    .input(z.object({ projectId: z.string() }))
    .mutation(async ({ ctx, input }) => {
      // Get default user
      const defaultUser = await ctx.db.user.findFirst({
        where: { clerkUID: "anonymous" },
      });
      if (!defaultUser) throw new Error("User not found");

      // Find project (no auth required)
      const project = await ctx.db.project.findFirst({
        where: { id: input.projectId },
      });
      if (!project) throw new Error("Project not found");

      // Delete related records first (Jobs, Analytics)
      await ctx.db.job.deleteMany({
        where: { projectId: input.projectId },
      });

      await ctx.db.analytics.deleteMany({
        where: { projectId: input.projectId },
      });

      // Delete the project
      await ctx.db.project.delete({ where: { id: input.projectId } });

      // Check if this was the last project, and if so, create a new "Untitled Project"
      const remainingProjects = await ctx.db.project.findMany({
        where: { userId: defaultUser.id },
      });

      if (remainingProjects.length === 0) {
        // Create a new "Untitled Project" with blank canvas
        const newProject = await ctx.db.project.create({
          data: {
            name: "Untitled Project",
            userId: defaultUser.id,
            nodes: JSON.stringify([]), // Empty array for blank canvas
            analytics: {
              create: {
                data: JSON.stringify({}),
              },
            },
          },
        });

        return {
          ...newProject,
          nodes: [],
          createdNewUntitled: true,
        };
      }

      return { deleted: true, createdNewUntitled: false };
    }),

  editProject: publicProcedure
    .input(
      z.object({
        projectId: z.string(),
        newName: z.string().min(1),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      // Get default user
      const defaultUser = await ctx.db.user.findFirst({
        where: { clerkUID: "anonymous" },
      });

      // Check if another project with the same name already exists
      const existingProject = await ctx.db.project.findFirst({
        where: {
          name: input.newName,
          userId: defaultUser?.id,
          NOT: { id: input.projectId },
        },
      });

      if (existingProject) {
        throw new TRPCError({
          code: "BAD_REQUEST",
          message: "A project with the same name already exists.",
        });
      }

      // Update project name
      return ctx.db.project.update({
        where: { id: input.projectId },
        data: { name: input.newName },
      });
    }),

  updateNodes: publicProcedure
    .input(
      z.object({
        projectId: z.string(),
        nodes: z.array(z.object({})), // Expect an array of objects (the nodes)
      }),
    )
    .mutation(async ({ ctx, input }) => {
      console.log("Project ID:", input.projectId);
      console.log("Received Nodes:", input.nodes);

      try {
        // Find project (no auth required)
        const project = await ctx.db.project.findFirst({
          where: {
            id: input.projectId,
          },
        });

        if (!project) {
          console.error("Project not found. Project ID:", input.projectId);
          throw new TRPCError({
            code: "NOT_FOUND",
            message: "Project not found",
          });
        }

        console.log("Project found:", project);

        // Update nodes with the new JSON data (convert to string for SQLite)
        const updatedProject = await ctx.db.project.update({
          where: { id: input.projectId },
          data: {
            nodes: JSON.stringify(input.nodes), // Store as JSON string for SQLite
          },
        });

        console.log("Project updated with new nodes:", updatedProject);

        // Parse JSON string back to object
        return {
          ...updatedProject,
          nodes: typeof updatedProject.nodes === 'string' ? JSON.parse(updatedProject.nodes) : updatedProject.nodes,
        };
      } catch (error) {
        // If database error, just return success (nodes stored in localStorage anyway)
        console.warn("Database error updating nodes, using localStorage:", error);
        return {
          id: input.projectId,
          name: "Untitled Project",
          nodes: input.nodes,
          createdAt: new Date(),
          updatedAt: new Date(),
        };
      }
    }),
});
