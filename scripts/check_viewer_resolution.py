"""Verify native high-resolution frames and interaction quality recovery."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

out=Path(__file__).resolve().parents[1]/"outputs/line_drawing"
errors=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path="C:/Program Files/Google/Chrome/Application/chrome.exe",headless=True)
    page=browser.new_page(viewport={"width":1536,"height":1050},device_scale_factor=2)
    page.on("pageerror",lambda e:errors.append(str(e)))
    page.goto("http://127.0.0.1:17865",wait_until="networkidle")
    def wait():page.wait_for_function("stats && !busy && !dirty && renderedGeneration===requestGeneration",timeout=60000)
    wait();assert page.evaluate("canvas.width")==1536
    page.locator('.tabs [data-mode=lines]').click();wait()
    page.screenshot(path=str(out/"viewer_highres.png"))
    box=page.locator('#viewport').bounding_box();x=box['x']+box['width']/2;y=box['y']+box['height']/2
    page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+50,y+10,steps=4)
    page.wait_for_function("stats.resolution===768 && !busy",timeout=60000)
    page.mouse.up();wait();assert page.evaluate("canvas.width")==1536
    page.locator('#resetCamera').click();wait()
    page.select_option('#resolution','2048');wait();assert page.evaluate("canvas.width")==2048
    with page.expect_download() as d:page.locator('#save').click()
    d.value.save_as(str(out/'lines_2048.png'))
    high=page.evaluate('stats')
    page.locator('#rotate').click();page.wait_for_function("auto && stats.resolution===768",timeout=60000)
    page.locator('#rotate').click();wait();assert page.evaluate('canvas.width')==2048
    page.select_option('#resolution','1536');page.locator('#resetCamera').click();wait()
    assert not errors,errors
    report=dict(errors=errors,high=high,checks=['default1536','drag768','release1536','export2048','auto768','stop2048','highDPI'])
    (out/'resolution_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));browser.close()
