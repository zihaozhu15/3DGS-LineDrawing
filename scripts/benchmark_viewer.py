"""Measure sequential API updates, excluding browser presentation."""
import json
import statistics
import time
from pathlib import Path
import requests

s=requests.Session();s.trust_env=False
url='http://127.0.0.1:17865'
info=s.get(url+'/api/info',timeout=10).json()
pose=info['presets'][0].copy()
report=[]
for resolution in (768,1536,2048):
    samples=[]
    for i in range(4):
        pose['yaw']+=.01
        start=time.perf_counter()
        r=s.post(url+'/api/render',json=dict(pose=pose,options=info['defaults'],resolution=resolution),timeout=30)
        r.raise_for_status();data=r.json()
        elapsed=(time.perf_counter()-start)*1000
        if i:
            samples.append(dict(gpu_ms=data['stats']['raster_ms'],server_ms=data['stats']['server_ms'],request_ms=elapsed,payload_kb=len(r.content)/1024))
    report.append(dict(resolution=resolution,**{key:round(statistics.mean(v[key] for v in samples),1) for key in samples[0]}))
print(json.dumps(report,indent=2))
(Path(__file__).resolve().parents[1]/'outputs/line_drawing/performance.json').write_text(json.dumps(report,indent=2))
