"""Exercise the actual generated offline page in an isolated Chromium profile."""
import asyncio
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request
import pytest
from .test_core import make_kit

CHROME = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
pytestmark = pytest.mark.skipif(not CHROME or not importlib.util.find_spec("websockets"),
                               reason="Chrome/Chromium and websockets needed for offline browser test")


async def exercise(ws, kit):
    import websockets
    async with websockets.connect(ws, max_size=16*1024*1024) as connection:
        seq = 0
        async def call(method, params=None):
            nonlocal seq
            seq += 1
            await connection.send(json.dumps(dict(id=seq, method=method, params=params or {})))
            while True:
                result = json.loads(await connection.recv())
                if result.get("id") == seq:
                    assert "error" not in result, result
                    return result.get("result", {})
        async def evaluate(source):
            result = await call("Runtime.evaluate", dict(expression=source, awaitPromise=True, returnByValue=True))
            assert "exceptionDetails" not in result, result
            return result.get("result", {}).get("value")
        async def wait(source):
            for _ in range(100):
                if await evaluate(source):
                    return
                await asyncio.sleep(.1)
            raise AssertionError(source)
        await call("Page.enable")
        await call("Page.navigate", dict(url=(kit/"开始标注.html").as_uri()))
        await wait("typeof loaded!=='undefined'&&loaded&&classes.length===3")
        assert await evaluate("canvas.width===64&&canvas.height===64&&rows.length===6")
        await evaluate("$('#complete').click()")
        assert await evaluate("!current().reviewed&&$('#message').textContent.includes('全部目标')")
        await evaluate("canvas.onclick({clientX:canvas.getBoundingClientRect().left+canvas.getBoundingClientRect().width/2,clientY:canvas.getBoundingClientRect().top+canvas.getBoundingClientRect().height/2})")
        assert await evaluate("JSON.stringify(current().objects.coin.point_xy)==='[32,32]'")
        await evaluate("selected=1;setStatus('not_visible');selected=2;setStatus('ignore');$('#complete').click()")
        await wait("loaded&&position===1")
        assert await evaluate("annotations[rows[0].id].reviewed")
        await evaluate("URL.createObjectURL=b=>{window.__blob=b;return 'blob:smoke'};HTMLAnchorElement.prototype.click=function(){};$('#export').click()")
        await evaluate("(async()=>{window.__export=JSON.parse(await window.__blob.text())})()")
        assert await evaluate("__export.images[0].reviewed&&__export.images[0].objects.marker.status==='ignore'")
        await evaluate("(async()=>{annotations={};await $('#importFile').onchange({target:{files:[{text:async()=>JSON.stringify(__export)}],value:'fake'}})})()")
        assert await evaluate("$('#message').textContent==='导入成功。'&&annotations[rows[0].id].reviewed")
        await evaluate("(async()=>{const bad=JSON.parse(JSON.stringify(__export));bad.kit_id='other';await $('#importFile').onchange({target:{files:[{text:async()=>JSON.stringify(bad)}],value:'fake'}})})()")
        assert await evaluate("$('#message').textContent.includes('导入失败')")
        await evaluate("window.__old=true")
        await call("Page.reload")
        await wait("!window.__old&&typeof loaded!=='undefined'&&loaded&&position===1")
        assert await evaluate("annotations[rows[0].id].reviewed")


def test_offline_annotation_roundtrip(tmp_path):
    kit = make_kit(tmp_path)
    profile = tmp_path / "chrome_profile"
    proc = subprocess.Popen([CHROME, "--headless=new", "--no-sandbox", "--disable-gpu", "--no-first-run",
                             "--no-default-browser-check", "--remote-debugging-port=0",
                             "--user-data-dir="+str(profile), "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        port_file = profile / "DevToolsActivePort"
        for _ in range(100):
            if port_file.exists():
                break
            time.sleep(.1)
        port = int(port_file.read_text().splitlines()[0])
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{port}/json/list") as response:
            page = next(p for p in json.load(response) if p["type"] == "page")
        asyncio.run(exercise(page["webSocketDebuggerUrl"], kit))
    finally:
        proc.terminate()
        proc.wait(timeout=15)
