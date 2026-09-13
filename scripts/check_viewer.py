"""Exercise the unified browser UI and verify its fast display path."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/"outputs/line_drawing";out.mkdir(exist_ok=True)
errors=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path="C:/Program Files/Google/Chrome/Application/chrome.exe",headless=True)
    page=browser.new_page(viewport={"width":1536,"height":1050},device_scale_factor=1)
    page.on("pageerror",lambda e:errors.append(str(e)))
    page.goto("http://127.0.0.1:17865",wait_until="networkidle")
    wait=lambda:page.wait_for_function("lastBitmap && !busy && !dirty && renderedGeneration===requestGeneration",timeout=60000)
    wait();assert page.locator("#error").is_hidden()
    previews=page.locator(".thumb img").evaluate_all("els=>els.map(e=>e.src)")
    assert len(previews)==6 and all(len(src)>100 for src in previews)
    page.locator("#topkOnly").click();wait()
    assert page.evaluate("options.enabled.join(',')")=='false,false,false,false,true'
    box=page.locator("#viewport").bounding_box();x=box['x']+box['width']/2;y=box['y']+box['height']/2
    page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+90,y+20,steps=10);page.mouse.up();wait()
    assert previews==page.locator(".thumb img").evaluate_all("els=>els.map(e=>e.src)"),"Fixed previews changed"
    page.locator("#rotate").click();page.wait_for_timeout(2200)
    fps=int(page.locator("#fps").inner_text());timing=page.locator("#frameTiming").inner_text()
    page.locator("#rotate").click();wait();assert fps>0 and 'GPU' in timing
    page.locator("#benchmarkFields").click()
    page.wait_for_function("document.querySelector('#benchmarkResult').textContent.includes('FPS')",timeout=60000)
    page.select_option("#mode","depth");wait();assert page.locator("#error").is_hidden()
    page.select_option("#mode","lines");wait()
    page.screenshot(path=str(out/"viewer_final.png"),full_page=True)
    report=dict(errors=errors,fps=fps,timing=timing,benchmark=page.locator("#benchmarkResult").inner_text(),checks=["fixed_previews","topk_only","binary_main_frame","prominent_fps","gpu_benchmark","diagnostic_on_demand"])
    browser.close()
assert not errors,errors
(out/"browser_validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(report,ensure_ascii=False,indent=2))
