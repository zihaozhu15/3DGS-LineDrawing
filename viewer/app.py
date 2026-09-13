"""Local interactive viewer. Run via scripts/start_viewer.cmd."""
import argparse
import threading
import traceback
from flask import Flask, jsonify, request, send_from_directory, Response
from renderer import LineRenderer

app=Flask(__name__,static_folder="static")
app.config["MAX_CONTENT_LENGTH"]=32768
renderer=None
lock=threading.Lock()

@app.get("/")
def index():return send_from_directory(app.static_folder,"index.html")

@app.post("/api/fast")
def fast():
    try:
        with lock:payload,stats=renderer.fast_draw(request.get_json())
        return Response(payload,mimetype="application/octet-stream",headers={
            'X-Resolution':str(stats['resolution']),'X-GPU-Ms':f"{stats['gpu_ms']:.3f}",
            'X-Server-Ms':f"{stats['server_ms']:.3f}",'X-Channels':str(stats['channels']),'Cache-Control':'no-store'})
    except (ValueError,AssertionError,KeyError,TypeError) as e:return jsonify(error=str(e) or 'Invalid parameters'),400
    except Exception:
        traceback.print_exc();return jsonify(error='Fast render failed; see viewer-error.log'),500

@app.post("/api/benchmark")
def benchmark():
    with lock:return jsonify(renderer.benchmark(request.get_json()))

@app.get("/api/info")
def info():return jsonify(renderer.info())

@app.get("/api/previews")
def previews():
    with lock:return jsonify(renderer.previews())

@app.post("/api/render")
def render():
    try:
        with lock:return jsonify(renderer.draw(request.get_json()))
    except (ValueError,AssertionError,KeyError,TypeError) as e:return jsonify(error=str(e) or "参数无效"),400
    except Exception:
        traceback.print_exc();return jsonify(error="渲染失败，请查看 logs/viewer.log"),500

@app.get("/api/pixel")
def pixel():
    try:
        with lock:return jsonify(renderer.inspect(request.args["x"],request.args["y"],request.args["frame"]))
    except (ValueError,KeyError) as e:return jsonify(error=str(e)),409

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--port",type=int,default=17865)
    args=parser.parse_args()
    renderer=LineRenderer()
    print(f"Lego Line Studio ready at http://127.0.0.1:{args.port}",flush=True)
    app.run(host="127.0.0.1",port=args.port,threaded=True,debug=False,use_reloader=False)
