"""Playwright UI test harness for automated testing of LLM applications.

This module provides a skeleton for running Playwright tests on web-based LLM apps.
It includes headless browser setup, test execution, and result collection.

Usage:
    harness = PlaywrightHarness()
    results = harness.run_test("example_test", url="http://localhost:3000")
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PlaywrightHarness:
    """Harnesses Playwright for UI testing of LLM applications.

    Tests can be defined as async functions that interact with the browser.
    """

    def __init__(self, headless: bool = True, browser: str = "chromium"):
        self.headless = headless
        self.browser = browser
        self._playwright = None
        self._browser = None

    async def _setup(self):
        """Initialize Playwright and browser."""
        try:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            if self.browser == "chromium":
                self._browser = await self._playwright.chromium.launch(
                    headless=self.headless
                )
            elif self.browser == "firefox":
                self._browser = await self._playwright.firefox.launch(
                    headless=self.headless
                )
            else:
                raise ValueError(f"Unsupported browser: {self.browser}")
        except Exception as e:
            logger.exception("Failed to setup Playwright: %s", e)
            raise

    async def _teardown(self):
        """Clean up browser and Playwright."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def run_test(self, test_name: str, url: str, **kwargs) -> Dict[str, Any]:
        """Run a specific test by name.

        Args:
            test_name: Name of the test method (e.g., "example_form_test")
            url: URL to test against
            **kwargs: Additional args passed to the test

        Returns:
            Dict with test results, including success, errors, and metrics.
        """
        await self._setup()
        try:
            page = await self._browser.new_page()
            await page.goto(url)

            # Dispatch to test method
            if test_name == "example_form_test":
                result = await self._example_form_test(page, **kwargs)
            else:
                raise ValueError(f"Unknown test: {test_name}")

            return {"success": True, "test": test_name, "result": result}
        except Exception as e:
            logger.exception("Test failed: %s", e)
            return {"success": False, "test": test_name, "error": str(e)}
        finally:
            await self._teardown()

    async def _example_form_test(self, page, **kwargs) -> Dict[str, Any]:
        """Example test: Fill a form and submit, check response.

        Assumes a simple HTML form at the URL.
        """
        # Wait for form elements
        await page.wait_for_selector("input[name='query']", timeout=5000)
        await page.fill("input[name='query']", "What is AI?")

        # Submit
        await page.click("button[type='submit']")

        # Wait for response
        await page.wait_for_selector("#response", timeout=10000)
        response_text = await page.inner_text("#response")

        # Basic checks
        has_response = len(response_text.strip()) > 0
        contains_ai = "AI" in response_text or "artificial" in response_text.lower()

        return {
            "response_length": len(response_text),
            "has_response": has_response,
            "contains_ai": contains_ai,
            "response_preview": response_text[:200] + "..."
            if len(response_text) > 200
            else response_text,
        }


async def run_example_test(url: str = "http://localhost:3000") -> Dict[str, Any]:
    """Convenience function to run the example test."""
    harness = PlaywrightHarness()
    return await harness.run_test("example_form_test", url=url)


# Synchronous wrapper for CLI
def run_playwright_demo(url: str = "http://localhost:3000") -> Dict[str, Any]:
    """Run the Playwright demo synchronously."""
    return asyncio.run(run_example_test(url))
