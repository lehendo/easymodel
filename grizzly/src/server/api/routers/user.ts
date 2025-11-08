import { z } from "zod";
import { createTRPCRouter, publicProcedure } from "../trpc";
import { TRPCError } from "@trpc/server";

export const userRouter = createTRPCRouter({
  // Sync User: Create a user in the DB if it doesn't exist
  syncUser: publicProcedure
    .input(z.object({ clerkUID: z.string() }))
    .mutation(async ({ ctx, input }) => {
      const { clerkUID } = input;

      // Check if user already exists
      const existingUser = await ctx.db.user.findUnique({
        where: { clerkUID },
      });

      if (existingUser) {
        return existingUser; // User already synced
      }

      // Create a new user
      const newUser = await ctx.db.user.create({
        data: {
          clerkUID,
        },
      });

      return newUser;
    }),

  // Optional: Update user information
  updateUser: publicProcedure
    .input(
      z.object({
        clerkUID: z.string(),
        name: z.string().optional(),
        email: z.string().email().optional(),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const { clerkUID, name, email } = input;

      const user = await ctx.db.user.update({
        where: { clerkUID },
        data: {
          ...(name && { name }),
          ...(email && { email }),
        },
      });

      return user;
    }),
});
