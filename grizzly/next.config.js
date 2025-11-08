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
  // Explicitly configure webpack to resolve @ alias
  webpack: (config) => {
    const srcPath = path.resolve(__dirname, "src");
    
    // Ensure resolve.alias exists
    config.resolve = config.resolve || {};
    config.resolve.alias = config.resolve.alias || {};
    
    // Add the @ alias, preserving existing aliases
    Object.assign(config.resolve.alias, {
      "@": srcPath,
      "@/": srcPath + "/",
    });
    
    console.log("Webpack alias configured:", config.resolve.alias["@"]);
    
    return config;
  },
};

export default config;
