/**
 * Run `build` or `dev` with `SKIP_ENV_VALIDATION` to skip env validation. This is especially useful
 * for Docker builds.
 */
import "./src/env.js";
import path from "path";

/** @type {import("next").NextConfig} */
const config = {
  typescript: {
    // ⚠️ Dangerously allow production builds to successfully complete even if
    // your project has type errors.
    ignoreBuildErrors: true,
  },
  eslint: {
    // Warning: This allows production builds to successfully complete even if
    // your project has ESLint errors.
    ignoreDuringBuilds: true,
  },
  // Explicitly configure webpack to resolve @ alias
  webpack: (config) => {
    // Use process.cwd() which is more reliable across environments
    const srcPath = path.join(process.cwd(), "src");
    
    // Ensure resolve.alias exists
    config.resolve = config.resolve || {};
    config.resolve.alias = config.resolve.alias || {};
    
    // Add the @ alias, preserving existing aliases
    Object.assign(config.resolve.alias, {
      "@": srcPath,
      "@/": path.join(process.cwd(), "src") + path.sep,
    });
    
    console.log("Webpack alias configured with cwd:", process.cwd(), "->", srcPath);
    
    return config;
  },
};

export default config;
