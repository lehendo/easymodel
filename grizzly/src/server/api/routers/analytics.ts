import { z } from "zod";
import { createTRPCRouter, publicProcedure } from "../trpc";

export const analyticsRouter = createTRPCRouter({
  // Fetch analytics data by project
  getAnalyticsByProject: publicProcedure
    .input(z.object({ projectId: z.string() }))
    .query(async ({ ctx, input }) => {
      // No auth required - just fetch analytics data
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
      // No auth required - just update analytics data
      return ctx.db.analytics.update({
        where: { projectId: input.projectId },
        data: { data: JSON.stringify(input.data) },
      });
    }),

  // Upsert analytics data (create or update)
  upsertAnalyticsData: publicProcedure
    .input(
      z.object({
        projectId: z.string(),
        data: z.record(z.any()), // JSON data for analytics
      }),
    )
    .mutation(async ({ ctx, input }) => {
      // Always use the incoming data (it contains all accumulated epochs from backend)
      // The backend sends the full accumulated analytics_data each time
      const dataString = JSON.stringify(input.data);
      
      // Check if analytics record exists
      const existingAnalytics = await ctx.db.analytics.findUnique({
        where: { projectId: input.projectId },
      });

      if (existingAnalytics) {
        // Update existing analytics with the new complete data
        return ctx.db.analytics.update({
          where: { projectId: input.projectId },
          data: { data: dataString },
        });
      } else {
        // Create new analytics record
        return ctx.db.analytics.create({
          data: {
            projectId: input.projectId,
            data: dataString,
          },
        });
      }
    }),
});
