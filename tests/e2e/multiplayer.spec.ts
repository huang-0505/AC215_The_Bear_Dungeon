/**
 * E2E: Multiplayer D&D Game — Full Lobby + Narration Round
 * =========================================================
 *
 * PREREQUISITES
 * -------------
 * 1. Bring up the full docker stack:
 *      docker compose up -d
 *    The stack must expose the Next.js frontend + backend API on port 8080
 *    via Nginx before you run this test.
 *
 * 2. Install Playwright (one-time, from the project root):
 *      npm install --save-dev @playwright/test
 *      npx playwright install chromium
 *
 *    Or with pnpm:
 *      pnpm add -D @playwright/test
 *      pnpm exec playwright install chromium
 *
 * HOW TO RUN
 * ----------
 *   npx playwright test tests/e2e/multiplayer.spec.ts
 *   pnpm exec playwright test tests/e2e/multiplayer.spec.ts
 *
 *   Add --headed to watch the browsers; --debug to step through.
 *
 * WHAT THIS TEST ASSERTS
 * ----------------------
 * 1. Host visits /multiplayer, fills identity fields (Display Name,
 *    Character Name, Class), opens the "Create Room" dialog, creates
 *    "Test Room E2E", and is auto-redirected to /multiplayer-game.
 * 2. Player2 visits /multiplayer in a separate browser context, fills
 *    their own identity, sees "Test Room E2E" in the lobby list, and
 *    clicks "Join Adventure" — also arriving at /multiplayer-game.
 * 3. Host clicks "Start Adventure" (no campaign selected → sandbox mode).
 * 4. Both players see the room transition from LOBBY → ACTIVE.
 * 5. Both players submit a narration-phase action.
 * 6. Both browsers receive a DM narrative response within 60 s
 *    (the orchestrator waits for all players or the 30 s timeout).
 *
 * TODO — STABILITY IMPROVEMENTS
 * ------------------------------
 * None of the interactive elements in this UI carry data-testid attributes.
 * The selectors below use roles, labels, and placeholder text which are
 * fragile against copy changes. Add the following attributes when you are
 * ready to harden the test suite:
 *
 *   Lobby page (/multiplayer):
 *     <Input id="playerName"     data-testid="input-player-name"     … />
 *     <Input id="characterName"  data-testid="input-character-name"  … />
 *     <SelectTrigger id="characterClass" data-testid="select-class"  … />
 *     <Button (Create Room)      data-testid="btn-create-room"       … />
 *     <Input id="roomName"       data-testid="input-room-name"       … />
 *     <Button (Create & Join)    data-testid="btn-confirm-create"    … />
 *     <Button (Join Adventure)   data-testid="btn-join-room"         … />
 *
 *   Game page (/multiplayer-game):
 *     <Button (Start Adventure)  data-testid="btn-start-adventure"   … />
 *     <Input (action text)       data-testid="input-action"          … />
 *     <Button (Submit)           data-testid="btn-submit-action"     … />
 *     Narrative container        data-testid="narrative-log"
 */

import { expect, test, type Page } from "@playwright/test"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Fill the identity card on the /multiplayer lobby page. */
async function fillIdentity(
  page: Page,
  playerName: string,
  characterName: string,
): Promise<void> {
  // Display Name input — identified by its HTML label ("Display Name") / placeholder ("Adventurer")
  await page.getByPlaceholder("Adventurer").fill(playerName)

  // Character Name input — placeholder "Eldrin"
  await page.getByPlaceholder("Eldrin").fill(characterName)

  // Class select stays at default "Fighter" — no change needed for the test.
  // If you need to change it, use:
  //   await page.getByRole("combobox").first().click()
  //   await page.getByRole("option", { name: "Wizard" }).click()
}

/** Wait until the URL contains /multiplayer-game */
async function waitForGamePage(page: Page): Promise<void> {
  await page.waitForURL(/\/multiplayer-game/, { timeout: 20_000 })
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

test.describe("Multiplayer flow — lobby creation, join, and narration round", () => {
  test("host creates room, player2 joins, both receive DM narrative after submitting actions", async ({
    browser,
  }) => {
    // Two isolated browser contexts simulate two different users.
    const hostContext = await browser.newContext()
    const player2Context = await browser.newContext()

    const hostPage = await hostContext.newPage()
    const player2Page = await player2Context.newPage()

    // ------------------------------------------------------------------
    // STEP 1 — Host: fill identity and create room
    // ------------------------------------------------------------------
    await hostPage.goto("/multiplayer")
    await expect(hostPage).toHaveTitle(/Arcane|DnD|Multiplayer|Next/i)

    await fillIdentity(hostPage, "HostPlayer", "Aria")

    // Open the Create Room dialog via the "Create Room" button in the header row.
    // The button has a "+" icon and the text "Create Room".
    await hostPage.getByRole("button", { name: /create room/i }).first().click()

    // Dialog should be visible
    await expect(
      hostPage.getByRole("dialog").getByText(/Create New Adventure Room/i),
    ).toBeVisible()

    // Fill room name
    await hostPage.getByPlaceholder("Epic Adventure Awaits").fill("Test Room E2E")

    // Submit the dialog — "Create & Join Room" button inside the dialog
    await hostPage
      .getByRole("dialog")
      .getByRole("button", { name: /create.*join room/i })
      .click()

    // Auto-redirect to /multiplayer-game
    await waitForGamePage(hostPage)

    // ------------------------------------------------------------------
    // STEP 2 — Player2: fill identity and join the room
    // ------------------------------------------------------------------
    await player2Page.goto("/multiplayer")

    await fillIdentity(player2Page, "Player2", "Borin")

    // Wait for the room list to render "Test Room E2E"
    await expect(
      player2Page.getByRole("heading", { name: "Test Room E2E" }),
    ).toBeVisible({ timeout: 15_000 })

    // Click the "Join Adventure" button on the "Test Room E2E" card.
    // The card containing the room name also holds the Join button.
    const roomCard = player2Page
      .locator(".grid")
      .filter({ hasText: "Test Room E2E" })
    await roomCard.getByRole("button", { name: /join adventure/i }).click()

    // Auto-redirect to /multiplayer-game
    await waitForGamePage(player2Page)

    // ------------------------------------------------------------------
    // STEP 3 — Both pages: confirm they are in the same room (LOBBY state)
    // ------------------------------------------------------------------
    // Each page should show "LOBBY" badge
    await expect(hostPage.getByText("LOBBY")).toBeVisible({ timeout: 10_000 })
    await expect(player2Page.getByText("LOBBY")).toBeVisible({ timeout: 10_000 })

    // Both pages should show "Test Room E2E" room name badge
    await expect(hostPage.getByText("Test Room E2E")).toBeVisible()
    await expect(player2Page.getByText("Test Room E2E")).toBeVisible()

    // ------------------------------------------------------------------
    // STEP 4 — Host: start the adventure (sandbox — no campaign selected)
    // ------------------------------------------------------------------
    // Only the host sees the "Start Adventure" button.
    await expect(
      hostPage.getByRole("button", { name: /start adventure/i }),
    ).toBeVisible({ timeout: 10_000 })

    await hostPage.getByRole("button", { name: /start adventure/i }).click()

    // ------------------------------------------------------------------
    // STEP 5 — Both pages: room transitions to ACTIVE
    // ------------------------------------------------------------------
    await expect(hostPage.getByText("ACTIVE")).toBeVisible({ timeout: 20_000 })
    await expect(player2Page.getByText("ACTIVE")).toBeVisible({ timeout: 20_000 })

    // After starting, the narrative log may show an opening scene.
    // We do not assert the exact text — just that it appears.
    // The DM prefix is "DM • R0" for the opening scene.
    // (Optional assertion — comment out if the orchestrator does not emit R0)
    // await expect(hostPage.locator("text=DM •")).toBeVisible({ timeout: 15_000 })

    // ------------------------------------------------------------------
    // STEP 6 — Both players: submit narration-phase actions
    // ------------------------------------------------------------------
    // The action input placeholder is "What do you do?" in narration phase.
    const hostInput = hostPage.getByPlaceholder("What do you do?")
    const player2Input = player2Page.getByPlaceholder("What do you do?")

    await expect(hostInput).toBeEnabled({ timeout: 10_000 })
    await expect(player2Input).toBeEnabled({ timeout: 10_000 })

    await hostInput.fill("I look around the room carefully.")
    await player2Input.fill("I draw my weapon and stand guard.")

    // Submit buttons — text is "Submit" in narration mode
    const hostSubmitBtn = hostPage.getByRole("button", { name: /^submit$/i })
    const player2SubmitBtn = player2Page.getByRole("button", { name: /^submit$/i })

    await hostSubmitBtn.click()
    await player2SubmitBtn.click()

    // ------------------------------------------------------------------
    // STEP 7 — Both pages: verify DM narrative arrives within 60 s
    // ------------------------------------------------------------------
    // After all players submit (or the 30 s orchestrator timeout fires),
    // a "DM • R1" entry should appear in both narrative logs.
    const dmNarrativePattern = /DM\s*•\s*R1/

    await expect(hostPage.locator("text=DM •").first()).toBeVisible({
      timeout: 60_000,
    })
    await expect(player2Page.locator("text=DM •").first()).toBeVisible({
      timeout: 60_000,
    })

    // Confirm the round-1 label is present specifically
    await expect(hostPage.getByText(dmNarrativePattern).first()).toBeVisible()
    await expect(player2Page.getByText(dmNarrativePattern).first()).toBeVisible()

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    await hostContext.close()
    await player2Context.close()
  })
})
