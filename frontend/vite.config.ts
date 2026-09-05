import { defineConfig, loadEnv } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  const usePolling = env.VITE_USE_POLLING === "true";

  return {
    plugins: [vue()],
    server: {
      ...(usePolling
        ? {
            watch: {
              interval: 1000,
              usePolling: true,
            },
          }
        : {}),
      proxy: {
        "/api": {
          target: "http://127.0.0.1:8000",
          changeOrigin: true,
        },
      },
    },
  };
});
