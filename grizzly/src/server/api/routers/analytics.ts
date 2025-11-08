import { z } from "zod";
import { createTRPCRouter, publicProcedure } from "../trpc";

export const analyticsRouter = createTRPCRouter({
  // Fetch analytics data by project
  getAnalyticsByProject: publicProcedure
    .input(z.object({ projectId: z.string() }))
    .query(async ({ ctx, input }) => {
      const { userId } = ctx.auth;
      if (!userId) throw new Error("Unauthorized");

      // Verify project ownership
      const project = await ctx.db.project.findFirst({
        where: {
          id: input.projectId,
          user: { clerkUID: userId },
        },
      });

      if (!project) throw new Error("Project not found or unauthorized");

      // Fetch analytics data for the project
      return ctx.db.analytics.findUnique({
        where: { projectId: input.projectId },
      });
    }),

  // Update analytics data
  updateAnalyticsData: publicProcedure
    .input(
      z.object({
        projectId: z.string(),
        data: z.record(z.any()), // JSON data for analytics
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const { userId } = ctx.auth;
      if (!userId) throw new Error("Unauthorized");

      // Verify project ownership
      const project = await ctx.db.project.findFirst({
        where: {
          id: input.projectId,
          user: { clerkUID: userId },
        },
      });

      if (!project) throw new Error("Project not found or unauthorized");

      // Update analytics data
      return ctx.db.analytics.update({
        where: { projectId: input.projectId },
        data: { data: input.data },
      });
    }),
});
