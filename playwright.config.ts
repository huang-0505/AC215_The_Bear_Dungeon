import { defineConfig, devices } from "@playwright/test"

/**
 * Playwright E2E configuration for DnD Master multiplayer flows.
 *
 * Prerequisites before running:
 *   docker compose up -d
 *
 * The webServer block documents the command but does NOT start it automatically
 * (reuseExistingServer: true means Playwright will use whatever is already
 * running on port 8080 and skip launching anything itself).
 *
 * Run tests:
 *   npm exec playwright test                         # all
 *   npm exec playwright test tests/e2e/multiplayer   # just multiplayer
 *   npm exec playwright test --headed                # watch the browser
 *   npm exec playwright test --debug                 # step-through debugger
 *   npm exec playwright show-report                  # open HTML report
 */
export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: false, // multiplayer tests share a live server — keep sequential
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  timeout: 90_000,
  expect: {
    timeout: 15_000,
  },
  reporter: [
    ["html", { outputFolder: "playwright-report", open: "never" }],
    ["junit", { outputFile: "playwright-report/results.xml" }],
    ["list"],
  ],
  use: {
    baseURL: "http://localhost:8080",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    video: "on-first-retry",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],

  // webServer documents how to start the stack but does NOT launch it.
  // Set reuseExistingServer: true so Playwright skips auto-launch and
  // connects to the already-running docker stack on port 8080.
  webServer: {
    command: "docker compose up",
    url: "http://localhost:8080",
    reuseExistingServer: true,
    timeout: 120_000,
  },
})
