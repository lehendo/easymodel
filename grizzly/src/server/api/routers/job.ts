import { z } from "zod";
import { createTRPCRouter, publicProcedure } from "@/server/api/trpc";
import { TRPCError } from "@trpc/server";
import { env } from "@/env";

export const jobRouter = createTRPCRouter({
  // Fetch all jobs (for anonymous users)
  getAllJobs: publicProcedure.query(async ({ ctx }) => {
    // Get default user
    const defaultUser = await ctx.db.user.findFirst({
      where: { clerkUID: "anonymous" },
    });

    if (!defaultUser) {
      return [];
    }

    return ctx.db.job.findMany({
      where: { clerkUID: defaultUser.clerkUID },
      orderBy: { createdAt: "desc" },
    });
  }),

  // Create a new job
  createJob: publicProcedure
    .input(
      z.object({
        type: z.enum(["ANALYTICS", "FINETUNING"]),
        projectId: z.string(),
        request: z.record(z.any()),
      }),
    )
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

      // Validate project exists
      const project = await ctx.db.project.findFirst({
        where: { id: input.projectId },
      });
      if (!project) throw new Error("Project not found");

      return ctx.db.job.create({
        data: {
          type: input.type,
          clerkUID: defaultUser.clerkUID,
          projectId: input.projectId,
          request: JSON.stringify(input.request), // Store as JSON string for SQLite
        },
      });
    }),

  // Execute finetuning job by calling easymodel backend
  executeFinetuning: publicProcedure
    .input(
      z.object({
        projectId: z.string(),
        modelName: z.string().min(1),
        datasets: z.array(z.string()).min(1),
        outputSpace: z.string().min(1),
        numEpochs: z.number().int().positive().default(1),
        batchSize: z.number().int().positive().default(8),
        maxLength: z.number().int().positive().default(128),
        subsetSize: z.number().int().positive().default(1000),
        taskType: z.enum(["generation", "classification", "seq2seq", "token_classification"]),
        textField: z.string().min(1),
        labelField: z.string().optional(),
      }),
    )
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

      // Validate project exists
      const project = await ctx.db.project.findFirst({
        where: { id: input.projectId },
      });
      if (!project) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "Project not found",
        });
      }

      // Create job record first
      const job = await ctx.db.job.create({
        data: {
          type: "FINETUNING",
          clerkUID: defaultUser.clerkUID,
          projectId: input.projectId,
          request: JSON.stringify(input), // Store as JSON string for SQLite
          status: "PENDING",
        },
      });

      try {
        // Call easymodel backend API
        const apiUrl = `${env.EASYMODEL_API_URL}/finetuning/`;
        const response = await fetch(apiUrl, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model_name: input.modelName,
            datasets: input.datasets,
            output_space: input.outputSpace,
            num_epochs: input.numEpochs,
            batch_size: input.batchSize,
            max_length: input.maxLength,
            subset_size: input.subsetSize,
            task_type: input.taskType,
            text_field: input.textField,
            label_field: input.labelField,
          }),
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`Backend error: ${response.status} - ${errorText}`);
        }

        const result = await response.json();

        // Update job status to running (training started)
        await ctx.db.job.update({
          where: { id: job.id },
          data: {
            status: "RUNNING",
            request: JSON.stringify({ ...input, backendJobId: result.job_id }), // Store backend job_id
          },
        });

        return {
          jobId: job.id,
          backendJobId: result.job_id, // Return backend job_id for SSE connection
          status: "RUNNING",
          message: result.message || "Fine-tuning started successfully",
        };
      } catch (error) {
        // Update job status to failed
        await ctx.db.job.update({
          where: { id: job.id },
          data: {
            status: "FAILED",
            request: JSON.stringify({ ...input, error: error instanceof Error ? error.message : String(error) }), // Store as JSON string for SQLite
          },
        });

        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: error instanceof Error ? error.message : "Failed to execute fine-tuning",
        });
      }
    }),

  // Cancel a running training job
  cancelTraining: publicProcedure
    .input(
      z.object({
        backendJobId: z.string().min(1),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      try {
        // Call easymodel backend cancel endpoint
        const apiUrl = `${env.EASYMODEL_API_URL}/finetuning/cancel/${input.backendJobId}`;
        const response = await fetch(apiUrl, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`Backend error: ${response.status} - ${errorText}`);
        }

        const result = await response.json();
        return {
          success: true,
          message: result.message || "Training cancellation requested",
          jobId: result.job_id,
          status: result.status || "cancelling", // Include status from backend
        };
      } catch (error) {
        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: error instanceof Error ? error.message : "Failed to cancel training",
        });
      }
    }),
});
