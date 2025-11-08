/**
 * YOU PROBABLY DON'T NEED TO EDIT THIS FILE, UNLESS:
 * 1. You want to modify request context (see Part 1).
 * 2. You want to create a new middleware or type of procedure (see Part 3).
 *
 * TL;DR - This is where all the tRPC server stuff is created and plugged in. The pieces you will
 * need to use are documented accordingly near the end.
 */
import { initTRPC, TRPCError } from "@trpc/server";
import superjson from "superjson";
import { ZodError } from "zod";

import { db } from "../db"; // Prisma DB import
import { NextRequest } from "next/server";

/**
 * 1. CONTEXT
 *
 * This section defines the "contexts" that are available in the backend API.
 *
 * These allow you to access things when processing a request, like the database, the session, etc.
 *
 * This helper generates the "internals" for a tRPC context. The API handler and RSC clients each
 * wrap this and provide the required context.
 *
 * @see https://trpc.io/docs/server/context
 */
export const createTRPCContext = async (opts: { req: NextRequest }) => {
  // Try to create or get default project, but handle database errors gracefully
  let defaultProjectId: string | null = null;
  
  try {
    // Get or create default user
    let defaultUser = await db.user.findFirst({
      where: { clerkUID: "anonymous" },
    });

    if (!defaultUser) {
      defaultUser = await db.user.create({
        data: {
          clerkUID: "anonymous",
          email: "anonymous@easymodel.local",
        },
      });
    }

    // Check if any projects exist
    const projects = await db.project.findMany({
      where: { userId: defaultUser.id },
    });

    // If no projects exist, create an "Untitled Project"
    if (projects.length === 0) {
      const untitledProject = await db.project.create({
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
      defaultProjectId = untitledProject.id;
    } else {
      // Use the first project as default
      defaultProjectId = projects[0].id;
    }
  } catch (error) {
    // Database not available - use in-memory mode
    console.warn("Database not available, using in-memory mode:", error);
    defaultProjectId = "untitled-project-in-memory";
  }

  return {
    db, // Database access via Prisma
    defaultProjectId, // Default project ID for anonymous users
  };
};

/**
 * 2. INITIALIZATION
 *
 * This is where the tRPC API is initialized, connecting the context and transformer. We also parse
 * ZodErrors so that you get typesafety on the frontend if your procedure fails due to validation
 * errors on the backend.
 */
const t = initTRPC.context<typeof createTRPCContext>().create({
  transformer: superjson,
  errorFormatter({ shape, error }) {
    return {
      ...shape,
      data: {
        ...shape.data,
        zodError:
          error.cause instanceof ZodError ? error.cause.flatten() : null,
      },
    };
  },
});

/**
 * Create a server-side caller.
 *
 * @see https://trpc.io/docs/server/server-side-calls
 */
export const createCallerFactory = t.createCallerFactory;

/**
 * 3. ROUTER & PROCEDURE (THE IMPORTANT BIT)
 *
 * These are the pieces you use to build your tRPC API. You should import these a lot in the
 * "/src/server/api/routers" directory.
 */

/**
 * This is how you create new routers and sub-routers in your tRPC API.
 *
 * @see https://trpc.io/docs/router
 */
export const createTRPCRouter = t.router;

/**
 * Middleware for timing procedure execution and adding an artificial delay in development.
 *
 * You can remove this if you don't like it, but it can help catch unwanted waterfalls by simulating
 * network latency that would occur in production but not in local development.
 */
const timingMiddleware = t.middleware(async ({ next, path }) => {
  const start = Date.now();

  if (t._config.isDev) {
    // artificial delay in dev
    const waitMs = Math.floor(Math.random() * 400) + 100;
    await new Promise((resolve) => setTimeout(resolve, waitMs));
  }

  const result = await next();

  const end = Date.now();
  console.log(`[TRPC] ${path} took ${end - start}ms to execute`);

  return result;
});

/**
 * Public (unauthenticated) procedure
 *
 * This is the base piece you use to build new queries and mutations on your tRPC API. It does not
 * guarantee that a user querying is authorized, but you can still access user session data if they
 * are logged in.
 */
export const publicProcedure = t.procedure.use(timingMiddleware);

// No authentication required - all procedures are public
export const protectedProcedure = publicProcedure;
