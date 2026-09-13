"""Check CUDA feature fields against an independent CPU reference and real render."""
import base64
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"viewer"))
from renderer import LineRenderer,DEFAULTS
from diff_gaussian_rasterization import _C

def reference(rgb,alpha,depth,normal,ids,weights,gate):
    _,H,W=rgb.shape;out=np.zeros((5,H,W),np.float32)
    for y in range(H):
        for x in range(W):
            for yy,xx in [(y,x-1),(y,x+1),(y-1,x),(y+1,x)]:
                if not (0<=xx<W and 0<=yy<H):continue
                a,b=alpha[0,y,x],alpha[0,yy,xx]
                delta=np.zeros(5)
                delta[0]=abs(a-b)
                if min(a,b)>=gate:
                    d,e=depth[0,y,x],depth[0,yy,xx]
                    delta[1]=abs(d-e)/max(d,e) if min(d,e)>0 else 0
                    n,m=normal[:,y,x],normal[:,yy,xx]
                    if np.linalg.norm(n)*np.linalg.norm(m)>1e-8:delta[2]=1-np.clip(np.dot(n,m)/(np.linalg.norm(n)*np.linalg.norm(m)),-1,1)
                    delta[3]=abs(np.dot([.2126,.7152,.0722],rgb[:,y,x]-rgb[:,yy,xx]))
                    u={int(i):float(v) for i,v in zip(ids[y,x],weights[y,x]) if i>=0}
                    v={int(i):float(w) for i,w in zip(ids[yy,xx],weights[yy,xx]) if i>=0}
                    su,sv=sum(u.values()),sum(v.values())
                    if su>0 and sv>0:delta[4]=.5*sum(abs(u.get(i,0)/su-v.get(i,0)/sv) for i in u.keys()|v.keys())
                out[:,y,x]=np.maximum(out[:,y,x],delta)
    return out

torch.manual_seed(7)
rgb=torch.rand(3,4,5);alpha=torch.rand(1,4,5);depth=torch.rand(1,4,5)+1
normal=torch.randn(3,4,5);ids=torch.arange(4,dtype=torch.int32).expand(4,5,4).clone()
weights=torch.rand(4,5,4)
# Same distribution with a different ranking, and a boundary containing a new ID.
ids[1,2]=ids[1,1].flip(0);weights[1,2]=weights[1,1].flip(0)
ids[2,2,0]=99;alpha[0,1,1:3]=1
args=[rgb,alpha,depth,normal,ids,weights]
expected=reference(*(t.numpy() for t in args),.5)
actual=_C.feature_fields(*(t.cuda().contiguous() for t in args),.5).cpu().numpy()
np.testing.assert_allclose(actual,expected,atol=2e-6)
renderer=LineRenderer()
out=ROOT/"outputs/line_drawing";out.mkdir(exist_ok=True)
report={"cpu_cuda_max_error":float(abs(actual-expected).max())}
for mode in ("rgb","lines","composite","edge_topk","depth","normal"):
    result=renderer.draw(dict(pose=renderer.presets[0],resolution=800,options={**DEFAULTS,"mode":mode}))
    (out/f"{mode}.png").write_bytes(base64.b64decode(result["image"]))
    assert np.isfinite(result["stats"]["server_ms"])
    report[mode]=result["stats"]
c=renderer.cache
assert (c["weights"][...,1:]<=c["weights"][...,:-1]+1e-7).all()
assert (c["weights"].sum(-1)<=c["alpha"][0]+2e-4).all()
assert c["ids"].max()<len(renderer.model.get_xyz)
assert c["ids"].min()>=-1
assert (c["raw"]>=0).all() and torch.isfinite(c["raw"]).all()
probe=renderer.inspect(400,400,renderer.frame)
assert "topk" in probe["p"]
(out/"validation.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
