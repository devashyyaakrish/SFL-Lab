#!/usr/bin/env python3
"""Exercise the running dashboard with real requests and system Chromium.

Run `.venv/bin/python scripts/verify_browser.py` after starting the service.
Use --browser to select another installed Chromium-compatible executable.
No fake responses or fabricated experiments are injected into the page.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import httpx
from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--browser", default=shutil.which("chromium") or shutil.which("google-chrome"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/browser"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checks, errors = [], []
    report = {"status": "running", "checks": checks, "errors": errors, "browser": args.browser}
    try:
        with httpx.Client(base_url=args.base_url, timeout=30) as client, sync_playwright() as p:
            browser = p.chromium.launch(executable_path=args.browser, headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
            page.set_default_timeout(15000)
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(args.base_url, wait_until="networkidle")
            expect(page.get_by_text("Engine connected", exact=True)).to_be_visible()
            expect(page.get_by_label("Model architecture", exact=True)).to_be_visible()
            assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
            page.screenshot(path=str(args.output / "desktop-setup.png"), full_page=True)
            checks.append("Desktop loads real API configuration without page errors or horizontal overflow")

            formula = page.get_by_label("Candidate mathematical formula")
            formula.fill("__import__('os')")
            page.get_by_role("button", name="Validate candidate", exact=True).click()
            expect(page.get_by_text("Changes required", exact=True)).to_be_visible()
            page.get_by_label("Load a formula template").select_option("gradient_clip")
            formula.click()
            formula.press("End")
            keyboard = page.get_by_label("Mathematical keyboard categories")
            keyboard.get_by_role("tab", name="Greek", exact=True).click()
            page.locator(".keyboard-keys button").first.click()
            assert formula.input_value() != "clip(G, C_G)"
            page.get_by_label("Load a formula template").select_option("gradient_clip")
            checks.append("Formula templates, mathematical keyboard and unsafe-expression rejection work")

            page.get_by_role("button", name="Algorithm", exact=True).click()
            page.get_by_role("button", name="Validate candidate", exact=True).click()
            expect(page.get_by_text("Ready for experiment", exact=True)).to_be_visible()
            page.get_by_label("Add algorithm operation").select_option("GRADIENT_CLIP")
            page.get_by_role("button", name="Validate candidate", exact=True).click()
            expect(page.get_by_text("Ready for experiment", exact=True)).to_be_visible()
            page.get_by_role("button", name="Remove GRADIENT_CLIP", exact=True).click()
            page.get_by_role("button", name="Formula", exact=True).click()
            page.get_by_label("Load a formula template").select_option("gradient_clip")
            checks.append("Controlled algorithm insertion, removal and validation work")

            name = "Browser verification: paired comparison"
            page.get_by_label("Experiment name", exact=True).fill(name)
            page.get_by_label("Model architecture", exact=True).select_option("tiny_cnn")
            page.get_by_label("Sample limit", exact=True).fill("80")
            page.get_by_label("Clients", exact=True).fill("2")
            page.get_by_label("Federated rounds", exact=True).fill("1")
            page.get_by_label("Random seeds · paired across variants", exact=True).fill("1, 2, 3")
            page.get_by_role("button", name="Compare", exact=True).click()
            expect(page.locator(".progress-title strong")).to_have_text(name, timeout=30000)
            expect(page.locator(".results-heading .badge")).to_have_text("Completed", timeout=240000)
            expect(page.get_by_role("img", name="Validation accuracy", exact=True)).to_be_visible()
            page.locator(".results-section").screenshot(path=str(args.output / "desktop-results.png"))
            checks.append("Comparison launched in UI completes real three-seed paired training and renders results")

            records = client.get("/api/experiments").json()
            full = next((e for e in records if e["status"] == "completed" and e["action"] == "full"), None)
            if full:
                page.get_by_role("button", name="Experiment history").click()
                page.locator(".history-item").filter(has_text=full["id"]).click()
                expect(page.locator(".experiment-id").first).to_have_text(full["id"])
            tabs = ["Overview", "Architecture", "Learning", "Privacy", "Federation", "Communication", "Scalability", "Network", "Parameters", "Ablation", "Statistics", "Attacks", "Final Report"]
            for tab in tabs:
                page.get_by_role("tablist", name="Experiment result views").get_by_role("tab", name=tab, exact=True).click()
                panel = page.get_by_role("tabpanel", name=tab, exact=True)
                expect(panel).to_be_visible()
                assert panel.inner_text().strip()
                assert not page.evaluate("document.documentElement.scrollWidth > innerWidth"), f"Overflow in {tab}"
            checks.append("All thirteen results tabs render stored evidence without page errors or overflow")

            if full:
                page.get_by_role("tablist", name="Experiment result views").get_by_role("tab", name="Network", exact=True).click()
                expect(page.get_by_role("img", name="Network cost heatmap")).to_be_visible()
                page.get_by_label("Network heatmap variant").select_option("baseline")
                page.get_by_label("Network heatmap metric").select_option("energy_joules")
                page.get_by_role("tablist", name="Experiment result views").get_by_role("tab", name="Attacks", exact=True).click()
                assert page.locator(".attack-card img").count() >= 11
                checks.append("Network heatmap controls and real reconstruction images render")

            page.get_by_role("tablist", name="Experiment result views").get_by_role("tab", name="Overview", exact=True).click()
            for format in ("PNG", "SVG"):
                with page.expect_download() as download:
                    page.locator(".chart-exports").first.get_by_role("button", name=format, exact=True).click()
                artifact = download.value
                path = args.output / artifact.suggested_filename
                artifact.save_as(path)
                assert path.stat().st_size > 100
            page.get_by_role("tablist", name="Experiment result views").get_by_role("tab", name="Final Report", exact=True).click()
            with page.expect_download() as download:
                page.get_by_role("link", name="PDF", exact=True).click()
            path = args.output / download.value.suggested_filename
            download.value.save_as(path)
            assert path.read_bytes().startswith(b"%PDF")
            checks.append("PNG, SVG and PDF exports download valid artifacts through the UI")

            page.get_by_role("button", name="Open evidence", exact=True).click()
            expect(page.locator(".evidence-json")).to_contain_text('"seeds"')
            page.get_by_role("button", name="Results", exact=True).click()
            expect(page.locator(".evidence-json")).to_contain_text('"baseline"')
            page.get_by_role("button", name="Close evidence").click()
            page.get_by_role("button", name="Switch to light mode").click()
            assert page.locator("html").get_attribute("data-theme") == "light"
            page.get_by_role("tablist", name="Experiment result views").get_by_role("tab", name="Overview", exact=True).click()
            page.locator(".results-section").screenshot(path=str(args.output / "light-results.png"))
            checks.append("Raw evidence drawer and light/dark theme controls work")

            page.get_by_role("button", name="New experiment", exact=True).click()
            page.get_by_label("Model architecture", exact=True).select_option("resnet18")
            page.get_by_label("Federated rounds", exact=True).fill("100")
            page.get_by_label("Sample limit", exact=True).fill("240")
            page.get_by_role("button", name="Baseline", exact=True).click()
            expect(page.get_by_role("button", name="Cancel", exact=True)).to_be_visible(timeout=30000)
            page.get_by_role("button", name="Cancel", exact=True).click()
            expect(page.locator(".results-heading .badge")).to_have_text("Cancelled", timeout=30000)
            checks.append("UI cancellation retains a complete record and polls to terminal cancelled state")

            page.set_viewport_size({"width": 390, "height": 844})
            page.evaluate("window.scrollTo(0,0)")
            assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
            page.get_by_role("button", name="Open navigation").click()
            page.get_by_role("button", name="Experiment history").click()
            expect(page.get_by_text("Your research trail", exact=False)).to_be_visible()
            page.get_by_role("button", name="Close history").click()
            page.screenshot(path=str(args.output / "mobile-setup.png"), full_page=True)
            checks.append("390px mobile layout, navigation and history work without horizontal overflow")
            assert not errors, errors
            browser.close()
            report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (args.output / "verification.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
