import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  fullyParallel: false,
  use: {
    baseURL: "http://127.0.0.1:3100",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: [
    {
      command: "node e2e/mock-kava-api.mjs",
      url: "http://127.0.0.1:8100/readyz",
      reuseExistingServer: false,
    },
    {
      command: "NEXT_PUBLIC_KAVA_API_URL=http://127.0.0.1:8100 pnpm exec vinext dev --port 3100 --hostname 127.0.0.1",
      url: "http://127.0.0.1:3100",
      reuseExistingServer: false,
    },
  ],
});
