"""Inference-only RaDe-GS feature lines and freely orbiting camera."""
import base64
import io
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/RaDe-GS"))
from scene import GaussianModel
from utils.graphics_utils import getProjectionMatrix
from diff_gaussian_rasterization import _C

FIELDS = ["opacity", "depth", "normal", "color", "topk"]
DEFAULTS = dict(thresholds=[.12,.012,.14,.055,.38], weights=[1.,.9,.5,.35,.24],
                enabled=[True]*5, softness=.25, width=1, detail_floor=.22, foreground=.5, k=8, mode="composite", ink="#202731", paper="#ffffff")

def encode(tensor, size=None):
    array=(tensor.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
    im=Image.fromarray(array)
    if size: im.thumbnail((size,size),Image.Resampling.BILINEAR)
    stream=io.BytesIO();im.save(stream,format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")

def heat(v):
    v=v.clamp(0,1)
    return torch.stack([(1.5-(4*v-3).abs()).clamp(0,1),(1.5-(4*v-2).abs()).clamp(0,1),(1.5-(4*v-1).abs()).clamp(0,1)])

class LineRenderer:
    def __init__(self):
        torch.set_num_threads(4)
        self.model=GaussianModel(3)
        self.model.load_ply(str(ROOT/"outputs/lego_radegs/point_cloud/iteration_30000/point_cloud.ply"))
        with torch.inference_mode():
            self.fast_scales,self.fast_opacity=self.model.get_scaling_n_opacity_with_3D_filter
            self.fast_rotation=self.model.get_rotation
            self.fast_sh=self.model.get_features
            self.fast_empty=torch.empty(0,device="cuda")
            self.fast_bg=torch.ones(3,device="cuda")
        self.cache_key=None;self.frame=0;self.cache=None
        transforms=json.loads((ROOT/"data/nerf_synthetic/lego/transforms_test.json").read_text())
        self.fov=float(transforms["camera_angle_x"])
        self.presets=[]
        for idx,name in [(160,"海报近似视角"),(40,"铲斗正面"),(80,"履带侧面"),(120,"另一侧"),(0,"后方")]:
            pos=np.array(transforms["frames"][idx]["transform_matrix"])[:3,3]
            radius=float(np.linalg.norm(pos))
            self.presets.append(dict(name=name,yaw=float(math.atan2(pos[1],pos[0])),pitch=float(math.asin(pos[2]/radius)),radius=radius,target=[0.,0.,0.]))

    def info(self):
        return dict(gaussians=len(self.model.get_xyz),gpu=torch.cuda.get_device_name(),iteration=30000,
                    defaults=DEFAULTS,presets=self.presets,fov=self.fov,fields=FIELDS)

    @torch.inference_mode()
    def fast_frame(self,request):
        """Always compute a fresh frame; no diagnostics, CPU copies or image codecs."""
        w=int(request.get("resolution",800));assert w in (512,768,800,1024,1536,2048)
        mode=request.get("mode","composite");assert mode in ("rgb","lines","composite")
        opts={**DEFAULTS,**request.get("options",{})}
        enabled=[bool(v) for v in opts["enabled"]]
        if len(enabled)!=5:raise ValueError("enabled must contain five feature fields")
        cam=self.camera(request.get("pose",self.presets[0]),w,w)
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        result=_C.rasterize_gaussians(self.fast_bg,self.model.get_xyz,self.fast_empty,self.fast_opacity,self.fast_scales,self.fast_rotation,
            1.,self.fast_empty,cam.view,cam.proj,math.tan(self.fov/2),math.tan(cam.fy/2),0.,w,w,self.fast_sh,
            self.model.active_sh_degree,cam.position,False,mode!="rgb" and (enabled[1] or enabled[2]),False)
        count,rgb,alpha,normal,depth,median,radii,geom,binning,img=result
        if mode=="rgb":output=rgb.clamp(0,1)
        else:
            k=int(opts['k']);assert k in (4,8)
            ids,weights,_=_C.extract_topk(geom,binning,img,len(self.model.get_xyz),count,w,w,k)
            raw=_C.feature_fields(rgb,alpha,depth,normal,ids,weights,float(opts['foreground']))
            scale=w/800.
            threshold=torch.tensor(opts['thresholds'],device="cuda")[:,None,None].clamp_min(1e-5)
            threshold=threshold/torch.tensor([scale,scale,scale**2,scale,scale],device="cuda")[:,None,None]
            softness=float(np.clip(opts['softness'],.05,1.))
            response=((raw-threshold)/(threshold*softness)+.5).clamp(0,1)
            response=response*response*(3-2*response)
            strength=torch.tensor(opts['weights'],device="cuda")[:,None,None].clamp(0,1)
            active=torch.tensor(enabled,device="cuda")[:,None,None]
            line=1-torch.prod(1-response*strength*active,dim=0)
            detail_floor=float(np.clip(opts['detail_floor'],0.,.7))
            line=((line-detail_floor)/(1-detail_floor)).clamp(0,1)
            width=int(np.clip(opts['width'],1,4))
            if width>1:line=F.max_pool2d(line[None,None],2*width-1,stride=1,padding=width-1)[0,0]
            def parse_color(value):
                if not isinstance(value,str) or len(value)!=7 or value[0]!='#':raise ValueError('invalid color')
                return torch.tensor([int(value[i:i+2],16)/255 for i in (1,3,5)],device='cuda')[:,None,None]
            ink,paper=parse_color(opts['ink']),parse_color(opts['paper'])
            base=paper.expand_as(rgb) if mode=="lines" else rgb.clamp(0,1)
            output=base*(1-line)+ink*line
        rgb8=(output.clamp(0,1).permute(1,2,0)*255).round().to(torch.uint8).contiguous()
        end.record()
        return rgb8,start,end

    @torch.inference_mode()
    def fast_draw(self,request):
        start_time=time.perf_counter()
        rgb8,start,end=self.fast_frame(request)
        payload=rgb8.cpu().numpy().tobytes()
        gpu_ms=start.elapsed_time(end)
        return payload,dict(resolution=rgb8.shape[0],channels=3,gpu_ms=gpu_ms,server_ms=(time.perf_counter()-start_time)*1000)

    @torch.inference_mode()
    def benchmark(self,request):
        gpu_times=[];pose=dict(request.get('pose',self.presets[0]));pose.pop('name',None)
        for i in range(33):
            pose['yaw']+=.005
            rgb8,start,end=self.fast_frame({**request,'pose':pose})
            end.synchronize()
            if i>=3:gpu_times.append(start.elapsed_time(end))
        ms=float(np.mean(gpu_times))
        enabled=request.get('options',{}).get('enabled',DEFAULTS['enabled'])
        active_fields=[name for name,on in zip(FIELDS,enabled) if on]
        return dict(samples=30,gpu_ms=ms,gpu_equivalent_fps=1000/ms,resolution=rgb8.shape[0],mode=request.get('mode','composite'),active_fields=active_fields,
                    note='CUDA event span: rasterization, required line processing and RGBA packing; excludes camera setup, readback, HTTP and display. No cached frames.')

    def camera(self,pose,w,h):
        yaw=float(pose["yaw"]);pitch=np.clip(float(pose["pitch"]),-1.45,1.45)
        radius=np.clip(float(pose["radius"]),1.,15.)
        target=np.clip(np.asarray(pose["target"],dtype=np.float64),-3,3)
        assert target.shape==(3,) and np.isfinite(target).all()
        assert np.isfinite([yaw,pitch,radius]).all()
        pos=target+radius*np.array([math.cos(pitch)*math.cos(yaw),math.cos(pitch)*math.sin(yaw),math.sin(pitch)])
        forward=target-pos;forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right)
        down=np.cross(forward,right)
        rotation=np.stack([right,down,forward],axis=0)
        w2c=np.eye(4,dtype=np.float32);w2c[:3,:3]=rotation;w2c[:3,3]=-rotation@pos
        view=torch.tensor(w2c.T.copy(),device="cuda")
        fy=2*math.atan(math.tan(self.fov/2)*h/w)
        proj=getProjectionMatrix(.01,100.,self.fov,fy).cuda().T
        return SimpleNamespace(view=view,proj=(view@proj).contiguous(),position=torch.tensor(pos,dtype=torch.float32,device="cuda"),fy=fy)

    @torch.inference_mode()
    def raster(self,pose,w,h,k):
        cam=self.camera(pose,w,h)
        scales,opacity=self.model.get_scaling_n_opacity_with_3D_filter
        empty=torch.empty(0,device="cuda")
        torch.cuda.synchronize();start=time.perf_counter()
        result=_C.rasterize_gaussians(torch.ones(3,device="cuda"),self.model.get_xyz,empty,opacity,scales,self.model.get_rotation,
            1.,empty,cam.view,cam.proj,math.tan(self.fov/2),math.tan(cam.fy/2),0.,h,w,self.model.get_features,
            self.model.active_sh_degree,cam.position,False,True,False)
        count,rgb,alpha,normal,depth,median,radii,geom,binning,img=result
        ids,weights,replay=_C.extract_topk(geom,binning,img,len(self.model.get_xyz),count,h,w,k)
        torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
        error=(replay-alpha).abs().max().item()
        if error>2e-4: raise RuntimeError(f"Top-k replay opacity mismatch: {error}")
        return dict(rgb=rgb,alpha=alpha,normal=normal,depth=depth,median=median,ids=ids,weights=weights,
                    raster_ms=elapsed,replay_error=error,visible=int((radii>0).sum()),raw=None,gate=None)

    @torch.inference_mode()
    def draw(self,request):
        start=time.perf_counter()
        w=int(request.get("resolution",1536));w=max(256,min(2048,w));h=w
        opts={**DEFAULTS,**request.get("options",{})}
        k=int(opts["k"]);assert k in (4,8)
        pose=request.get("pose",self.presets[0])
        key=json.dumps([pose,w,h,k],sort_keys=True)
        cached=key==self.cache_key
        if not cached:
            self.cache=self.raster(pose,w,h,k);self.cache_key=key
        c=self.cache
        foreground=float(np.clip(opts["foreground"],.05,.99))
        if c["gate"]!=foreground:
            c["raw"]=_C.feature_fields(c["rgb"],c["alpha"],c["depth"],c["normal"],c["ids"],c["weights"],foreground)
            c["gate"]=foreground
        threshold=torch.tensor(opts["thresholds"],dtype=torch.float32,device="cuda")[:,None,None].clamp_min(1e-5)
        # Thresholds are calibrated at 800 px. First differences scale with
        # pixel spacing; 1-dot(normal_p, normal_q) is quadratic for smooth normals.
        pixel_scale=w/800.
        threshold=threshold/torch.tensor([pixel_scale,pixel_scale,pixel_scale**2,pixel_scale,pixel_scale],device="cuda")[:,None,None]
        weight=torch.tensor(opts["weights"],dtype=torch.float32,device="cuda")[:,None,None].clamp(0,1)
        enabled=torch.tensor(opts["enabled"],device="cuda")[:,None,None]
        assert threshold.shape==(5,1,1) and weight.shape==(5,1,1) and enabled.shape==(5,1,1)
        softness=float(np.clip(opts["softness"],.05,1.))
        e=((c["raw"]-threshold)/(threshold*softness)+.5).clamp(0,1)
        e=e*e*(3-2*e)
        line=1-torch.prod(1-e*weight*enabled,dim=0)
        detail_floor=float(np.clip(opts["detail_floor"],0.,.7))
        line=((line-detail_floor)/(1-detail_floor)).clamp(0,1)
        width=int(np.clip(opts["width"],1,4))
        if width>1: line=F.max_pool2d(line[None,None],2*width-1,stride=1,padding=width-1)[0,0]
        def color(s):
            assert isinstance(s,str) and len(s)==7 and s[0]=="#"
            return torch.tensor([int(s[i:i+2],16)/255 for i in (1,3,5)],device="cuda")[:,None,None]
        ink,paper=color(opts["ink"]),color(opts["paper"])
        drawing=paper*(1-line)+ink*line
        composite=c["rgb"].clamp(0,1)*(1-line)+ink*line
        foreground_mask=c["alpha"][0]>.5
        depth=c["depth"][0]
        valid=depth[foreground_mask & (depth>0)]
        if valid.numel():
            lo,hi=torch.quantile(valid,torch.tensor([.02,.98],device="cuda"))
            dv=((depth-lo)/(hi-lo).clamp_min(1e-6)).clamp(0,1)
            depth_range=[lo.item(),hi.item()]
        else: dv=torch.zeros_like(depth);depth_range=[0,0]
        depth_image=torch.where(foreground_mask[None],heat(1-dv),paper)
        normal_image=torch.where(foreground_mask[None],(c["normal"]+1)*.5,paper)
        coverage=(c["weights"].sum(-1)/c["alpha"][0].clamp_min(1e-6)).clamp(0,1)
        id0=c["ids"][...,0].long().clamp_min(0)
        identity=torch.stack([((id0*mul+17)%255)/255 for mul in (37,73,151)])
        identity=torch.where(foreground_mask[None],identity,paper)
        displays=dict(rgb=c["rgb"],lines=drawing,composite=composite,depth=depth_image,normal=normal_image,
                      opacity=c["alpha"].repeat(3,1,1),coverage=torch.where(foreground_mask[None],heat(coverage),paper),ids=identity)
        for i,name in enumerate(FIELDS):displays["edge_"+name]=paper*(1-e[i])+ink*e[i]
        mode=opts["mode"]
        if mode not in displays:raise ValueError("Unknown display mode")
        self.frame+=1;c["line"]=line;c["response"]=e
        image=encode(displays[mode])
        thumbs={}
        if request.get("include_thumbnails",False):
            thumbs={name:encode(displays[name],144) for name in ("rgb","lines","depth","normal","opacity","edge_topk")}
        stats=dict(frame=self.frame,resolution=w,visible=c["visible"],raster_ms=round(c["raster_ms"],1),cached=cached,
                   coverage=round(coverage[foreground_mask].mean().item(),3) if foreground_mask.any() else 0,
                   line_fraction=round((line>.25).float().mean().item(),3),depth_range=depth_range,replay_error=c["replay_error"])
        stats["server_ms"]=round((time.perf_counter()-start)*1000,1)
        return dict(image=image,thumbnails=thumbs,stats=stats)

    @torch.inference_mode()
    def previews(self):
        request=dict(pose=self.presets[0],options=DEFAULTS,resolution=512,include_thumbnails=True)
        result=self.draw(request)
        return dict(thumbnails=result["thumbnails"],stats=result["stats"])

    @torch.inference_mode()
    def inspect(self,x,y,frame):
        if not self.cache or int(frame)!=self.frame:raise ValueError("视角已更新，请重新点击")
        c=self.cache;h,w=c["alpha"].shape[-2:];x=max(0,min(w-1,int(x)));y=max(0,min(h-1,int(y)))
        def pixel(x,y):
            weights=c["weights"][y,x];total=weights.sum().clamp_min(1e-8)
            return dict(x=x,y=y,rgb=c["rgb"][:,y,x].tolist(),alpha=c["alpha"][0,y,x].item(),depth=c["depth"][0,y,x].item(),
                normal=c["normal"][:,y,x].tolist(),edges=dict(zip(FIELDS,c["raw"][:,y,x].tolist())),
                topk=[dict(id=int(i),weight=float(v),normalized=float(v/total)) for i,v in zip(c["ids"][y,x],weights) if int(i)>=0])
        return dict(p=pixel(x,y),q=pixel(min(x+1,w-1),y))
