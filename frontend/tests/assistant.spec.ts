import { expect, test } from "@playwright/test";

test("Aurora keeps setup in drawers and requests audio without camera for a call", async ({ page }) => {
  await page.route("**/api/personas", route => route.fulfill({ json: { personas: [{ id: "alex", name: "Alex Morgan", language: "en" }] } }));
  await page.route("**/api/personas/alex/portrait", route => route.fulfill({ contentType: "image/png", body: Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4//8/AwAI/AL+XfGnGAAAAABJRU5ErkJggg==", "base64") }));
  await page.route("**/api/personas/script?language=en", route => route.fulfill({ json: { language: "en", text: "A short sample script.", rules: { portrait: "Face the camera.", idle_seconds: 8, voice_min_seconds: 15, voice_target_seconds: 25, voice_max_seconds: 45 }, consent: { version: "1", text: "I have permission." }, stock_voices: [] } }));
  await page.addInitScript(() => {
    (window as Window & { mediaRequests?: MediaStreamConstraints[] }).mediaRequests = [];
    navigator.mediaDevices.getUserMedia = async constraints => {
      if (constraints) (window as Window & { mediaRequests?: MediaStreamConstraints[] }).mediaRequests?.push(constraints);
      throw new DOMException("Microphone denied", "NotAllowedError");
    };
  });

  await page.goto("/assistant");
  await expect(page.getByText("Alex Morgan", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Transcript" })).toHaveCount(0);
  await page.getByRole("button", { name: "Show transcript" }).click();
  await expect(page.getByRole("heading", { name: "Transcript" })).toBeVisible();
  await page.getByRole("button", { name: "Open settings" }).click();
  await expect(page.getByRole("dialog", { name: "Assistant settings" })).toBeVisible();
  await expect(page.getByLabel("Conversation language")).toHaveValue("en");
  await page.getByRole("button", { name: "Close settings" }).click();
  await page.getByRole("button", { name: "Create a new face" }).click();
  await expect(page.getByRole("dialog", { name: "New face wizard" })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog", { name: "New face wizard" })).toHaveCount(0);

  await page.getByRole("button", { name: "Start conversation" }).click();
  await expect(page.locator(".aurora-error")).toContainText("Microphone denied");
  await expect(page.getByRole("button", { name: "Start conversation" })).toBeEnabled();
  const requests = await page.evaluate(() => (window as Window & { mediaRequests?: MediaStreamConstraints[] }).mediaRequests || []);
  expect(requests).toHaveLength(1);
  expect(requests[0].audio).toBeTruthy();
  expect(requests[0].video).toBe(false);
});
