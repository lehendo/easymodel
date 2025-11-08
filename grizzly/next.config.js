/**
 * Run `build` or `dev` with `SKIP_ENV_VALIDATION` to skip env validation. This is especially useful
 * for Docker builds.
 */
import "./src/env.js";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/** @type {import("next").NextConfig} */
const config = {
  webpack: (config, { isServer }) => {
    // Explicitly set the alias for @ path resolution
    // Use resolve to get absolute path, which works better in Vercel's build environment
    const srcPath = path.resolve(__dirname, "src");
    config.resolve.alias = {
      ...config.resolve.alias,
      "@": srcPath,
    };
    return config;
  },
};

export default config;
