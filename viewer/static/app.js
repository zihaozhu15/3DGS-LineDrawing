const $=s=>document.querySelector(s);
const names=['不透明度','深度','法线','色调','Top-k Gaussian'];
const keys=['opacity','depth','normal','color','topk'];
const limits=[[.01,.8,.01],[.001,.08,.001],[.01,1,.01],[.005,.3,.005],[.05,.95,.01]];
const thumbModes=['rgb','lines','depth','normal','opacity','edge_topk'];
const thumbNames=['RGB','融合线稿','深度','法线','不透明度','Top-k 线场'];
const fastModes=new Set(['rgb','lines','composite']);
const canvas=$('#viewport'),copy=o=>JSON.parse(JSON.stringify(o));
const gl=canvas.getContext('webgl2',{alpha:false,antialias:false,preserveDrawingBuffer:true});
if(!gl)throw new Error('浏览器不支持 WebGL2');
const vertex=`#version 300 es
in vec2 p;out vec2 uv;void main(){gl_Position=vec4(p,0.,1.);uv=vec2(p.x*.5+.5,1.-(p.y*.5+.5));}`;
const fragment=`#version 300 es
precision mediump float;uniform sampler2D frame;in vec2 uv;out vec4 color;void main(){color=vec4(texture(frame,uv).rgb,1.);}`;
function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s}
const program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vertex));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(program);gl.useProgram(program);
const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,3,-1,-1,3]),gl.STATIC_DRAW);const position=gl.getAttribLocation(program,'p');gl.enableVertexAttribArray(position);gl.vertexAttribPointer(position,2,gl.FLOAT,false,0,0);
const texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,texture);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);gl.pixelStorei(gl.UNPACK_ALIGNMENT,1);
function drawTexture(source,w,h,raw=false){if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}gl.viewport(0,0,w,h);gl.bindTexture(gl.TEXTURE_2D,texture);if(raw)gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB8,w,h,0,gl.RGB,gl.UNSIGNED_BYTE,source);else gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB8,gl.RGB,gl.UNSIGNED_BYTE,source);gl.drawArrays(gl.TRIANGLES,0,3)}
let info,options,pose,defaultPose,stats,busy=false,dirty=false,drag=null,auto=false,autoTimer;
let requestGeneration=0,renderedGeneration=-1,displayedFrameTimes=[],lastBitmap=null,pendingFrame=null,displayScheduled=false;

function updateFPS(){const now=performance.now();displayedFrameTimes=displayedFrameTimes.filter(t=>now-t<1000);$('#fps').textContent=displayedFrameTimes.length}
setInterval(updateFPS,250);

function buildControls(){
 $('#fields').innerHTML=names.map((name,i)=>`<div class="field" id="field${i}"><div class="field-heading"><label><input type="checkbox" data-enable="${i}" checked>${name}</label><span class="field-index">0${i+1}</span></div><label class="control-label">阈值 <output id="t${i}"></output></label><input aria-label="${name}阈值" type="range" data-threshold="${i}" min="${limits[i][0]}" max="${limits[i][1]}" step="${limits[i][2]}"><div class="strength-row">强度 <input aria-label="${name}强度" type="range" data-weight="${i}" min="0" max="1" step=".05"><output id="w${i}"></output></div></div>`).join('');
 $('#filmstrip').innerHTML=thumbModes.map((mode,i)=>`<button class="thumb" data-mode="${mode}"><img alt="${thumbNames[i]}固定预览"><span>${thumbNames[i]}</span></button>`).join('');
 document.querySelectorAll('[data-enable]').forEach(el=>el.onchange=()=>{options.enabled[+el.dataset.enable]=el.checked;syncControls();schedule()});
 document.querySelectorAll('[data-threshold]').forEach(el=>el.oninput=()=>{options.thresholds[+el.dataset.threshold]=+el.value;syncControls();schedule()});
 document.querySelectorAll('[data-weight]').forEach(el=>el.oninput=()=>{options.weights[+el.dataset.weight]=+el.value;syncControls();schedule()});
 document.querySelectorAll('[data-mode]').forEach(el=>el.onclick=()=>setMode(el.dataset.mode));
}
function syncControls(){
 keys.forEach((_,i)=>{const f=$(`#field${i}`);f.classList.toggle('disabled',!options.enabled[i]);f.querySelector('[data-enable]').checked=options.enabled[i];f.querySelector('[data-threshold]').value=options.thresholds[i];f.querySelector('[data-weight]').value=options.weights[i];$(`#t${i}`).textContent=options.thresholds[i].toFixed(i===1?3:2);$(`#w${i}`).textContent=options.weights[i].toFixed(2)});
 for(const name of ['width','softness','foreground','detail_floor','k','ink','paper'])$('#'+name).value=options[name];
 $('#widthValue').textContent=options.width+' px';$('#softnessValue').textContent=options.softness.toFixed(2);$('#foregroundValue').textContent=options.foreground.toFixed(2);$('#detail_floorValue').textContent=options.detail_floor.toFixed(2);
 $('#mode').value=options.mode;$('#viewLabel').textContent=$('#mode').selectedOptions[0].textContent;
 document.querySelectorAll('[data-mode]').forEach(el=>el.classList.toggle('active',el.dataset.mode===options.mode));
}
function setMode(mode){options.mode=mode;syncControls();schedule()}
function schedule(){dirty=true;requestGeneration++;if(!busy)requestAnimationFrame(render)}
function queueFrame(frame){pendingFrame=frame;if(!displayScheduled){displayScheduled=true;requestAnimationFrame(presentFrame)}}
function presentFrame(){
 const frame=pendingFrame;pendingFrame=null;
 if(frame){drawTexture(frame.source,frame.size,frame.size,frame.raw);lastBitmap=true;renderedGeneration=frame.generation;displayedFrameTimes.push(performance.now());updateFPS();const total=Math.round(performance.now()-frame.started);stats={...(stats||{}),resolution:frame.size,gpu_ms:frame.gpu,roundtrip_ms:total};$('#frameTiming').textContent=frame.raw?`GPU ${frame.gpu.toFixed(2)} ms · 总计 ${total} ms`:`诊断图 · 总计 ${total} ms`;$('#timing').textContent=frame.raw?`${frame.size} × ${frame.size} · GPU ${frame.gpu.toFixed(2)} ms · 完整帧 ${total} ms`:`${frame.size} × ${frame.size} · 按需诊断图 ${total} ms`;$('#poseLabel').textContent=`方位 ${(frame.pose.yaw*180/Math.PI).toFixed(0)}° / 仰角 ${(frame.pose.pitch*180/Math.PI).toFixed(0)}°`;$('#connection').textContent='GPU 已连接';$('#error').hidden=true}
 if(pendingFrame)requestAnimationFrame(presentFrame);else displayScheduled=false;
}

async function render(){
 if(busy||!dirty)return;
 busy=true;dirty=false;const generation=requestGeneration;
 $('#loading').hidden=false;$('#loading').classList.toggle('busy',!!lastBitmap);$('#loading').textContent=lastBitmap?'更新视图…':'正在生成首帧…';
 const started=performance.now(),resolution=(drag||auto)?Math.min(768,+$('#resolution').value):+$('#resolution').value;
 try{
  if(fastModes.has(options.mode)){
   const response=await fetch('/api/fast',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pose:copy(pose),options:copy(options),mode:options.mode,resolution})});
   if(!response.ok)throw new Error((await response.json()).error||'渲染失败');
   const bytes=await response.arrayBuffer(),size=+response.headers.get('X-Resolution'),channels=+response.headers.get('X-Channels');
   if(channels!==3||bytes.byteLength!==size*size*channels)throw new Error('帧数据长度不符');
   queueFrame({source:new Uint8Array(bytes),size,raw:true,gpu:+response.headers.get('X-GPU-Ms'),started,generation,pose:copy(pose)});
  }else{
   const response=await fetch('/api/render',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pose:copy(pose),options:copy(options),resolution,include_thumbnails:false})});
   const data=await response.json();if(!response.ok)throw new Error(data.error||'渲染失败');
   const image=new Image();image.src='data:image/png;base64,'+data.image;await image.decode();stats=data.stats;
   queueFrame({source:image,size:image.width,raw:false,gpu:stats.raster_ms,started,generation,pose:copy(pose)});
  }
 }catch(e){$('#error').textContent=e.message;$('#error').hidden=false;$('#connection').textContent='请求失败';stopAuto()}
 finally{busy=false;$('#loading').hidden=true;if(dirty)requestAnimationFrame(render);else if(auto)autoTimer=setTimeout(()=>{pose.yaw+=.035;schedule()},0)}
}

async function loadFixedPreviews(){
 const response=await fetch('/api/previews');const data=await response.json();if(!response.ok)throw new Error(data.error||'预览生成失败');
 thumbModes.forEach(mode=>$(`.thumb[data-mode="${mode}"] img`).src='data:image/png;base64,'+data.thumbnails[mode]);
 const s=data.stats;stats=s;$('#visible').textContent=s.visible.toLocaleString();$('#coverage').textContent=(s.coverage*100).toFixed(1)+'%';$('#lineFraction').textContent=(s.line_fraction*100).toFixed(1)+'%';
}
function stopAuto(){auto=false;clearTimeout(autoTimer);$('#rotate').classList.remove('active');$('#rotate').textContent='▷ 自动环绕'}
$('#rotate').onclick=()=>{if(auto){stopAuto();schedule()}else{auto=true;$('#rotate').classList.add('active');$('#rotate').textContent='Ⅱ 暂停环绕';schedule()}};
canvas.oncontextmenu=e=>e.preventDefault();
canvas.onpointerdown=e=>{stopAuto();canvas.setPointerCapture(e.pointerId);drag={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,button:e.button,shift:e.shiftKey,moved:false}};
canvas.onpointermove=e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.hypot(e.clientX-drag.startX,e.clientY-drag.startY)>3)drag.moved=true;if(drag.moved){if(drag.button===2||drag.shift){const scale=pose.radius*.0018,y=pose.yaw,p=pose.pitch,right=[-Math.sin(y),Math.cos(y),0],up=[-Math.sin(p)*Math.cos(y),-Math.sin(p)*Math.sin(y),Math.cos(p)];pose.target=pose.target.map((v,i)=>Math.max(-3,Math.min(3,v-dx*scale*right[i]+dy*scale*up[i])))}else{pose.yaw-=dx*.008;pose.pitch=Math.max(-1.45,Math.min(1.45,pose.pitch+dy*.008))}schedule()}drag.x=e.clientX;drag.y=e.clientY};
canvas.onpointerup=e=>{if(!drag)return;const moved=drag.moved;drag=null;if(moved)schedule();else inspect(e)};canvas.onpointercancel=()=>{drag=null;schedule()};
canvas.addEventListener('wheel',e=>{e.preventDefault();stopAuto();pose.radius=Math.max(1,Math.min(15,pose.radius*Math.exp(e.deltaY*.001)));schedule()},{passive:false});

async function inspect(e){
 if(busy)return;const rect=canvas.getBoundingClientRect(),x=Math.floor((e.clientX-rect.left)/rect.width*canvas.width),y=Math.floor((e.clientY-rect.top)/rect.height*canvas.height);
 $('#pixelCoord').textContent='读取…';
 try{const prep=await fetch('/api/render',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pose:copy(pose),options:{...copy(options),mode:'composite'},resolution:canvas.width,include_thumbnails:false})});const prepared=await prep.json();if(!prep.ok)throw new Error(prepared.error||'像素准备失败');const r=await fetch(`/api/pixel?x=${x}&y=${y}&frame=${prepared.stats.frame}`),d=await r.json();if(!r.ok)throw new Error(d.error);const p=d.p,q=d.q,all=[...new Set([...p.topk,...q.topk].map(t=>t.id))];$('#pixelCoord').textContent=`p (${p.x}, ${p.y})`;$('#probe').innerHTML=`<div class="probe-values">α ${p.alpha.toFixed(4)} · 深度 ${p.depth.toFixed(4)}<br>法线 [${p.normal.map(v=>v.toFixed(2)).join(', ')}]<br>亮度边缘 ${p.edges.color.toFixed(4)}<br>Top-k 差异 ${p.edges.topk.toFixed(4)}</div><p class="probe-title">按 Gaussian ID 对齐 · 归一化贡献</p><table class="probe-table"><thead><tr><th>ID</th><th>p</th><th>q →</th></tr></thead><tbody>${all.map(id=>{const a=p.topk.find(t=>t.id===id),b=q.topk.find(t=>t.id===id);return `<tr><td class="${a&&b?'match':''}">${id}</td><td>${a?(100*a.normalized).toFixed(1)+'%':'—'}</td><td>${b?(100*b.normalized).toFixed(1)+'%':'—'}</td></tr>`}).join('')}</tbody></table>`}catch(err){$('#pixelCoord').textContent=err.message}
}

for(const name of ['width','softness','foreground','detail_floor','k','ink','paper'])$('#'+name).oninput=e=>{options[name]=['ink','paper'].includes(name)?e.target.value:+e.target.value;syncControls();schedule()};
$('#mode').onchange=e=>setMode(e.target.value);$('#resolution').onchange=schedule;$('#camera').onchange=e=>{stopAuto();pose=copy(info.presets[+e.target.value]);defaultPose=copy(pose);schedule()};$('#resetCamera').onclick=()=>{stopAuto();pose=copy(defaultPose);schedule()};$('#resetStyle').onclick=()=>applyStyle('balanced');
$('#topkOnly').onclick=()=>{options.enabled=[false,false,false,false,true];options.mode='lines';syncControls();$('#benchmarkResult').textContent='';schedule()};
$('#benchmarkFields').onclick=async()=>{const button=$('#benchmarkFields');button.disabled=true;$('#benchmarkResult').textContent='预热并测量 30 帧…';try{const response=await fetch('/api/benchmark',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pose:copy(pose),mode:fastModes.has(options.mode)?options.mode:'lines',resolution:+$('#resolution').value,options:copy(options)})});const data=await response.json();if(!response.ok)throw new Error(data.error||'基准测试失败');const fields=data.active_fields.length?data.active_fields.join(' + '):'无';$('#benchmarkResult').textContent=`${fields}：GPU ${data.gpu_ms.toFixed(2)} ms ≈ ${data.gpu_equivalent_fps.toFixed(0)} FPS（不含传输与界面）`}catch(e){$('#benchmarkResult').textContent=e.message}finally{button.disabled=false}};
function applyStyle(name){const mode=options.mode;options=copy(info.defaults);options.mode=mode;if(name==='clean'){options.thresholds=[.16,.018,.32,.10,.55];options.weights=[1,.9,.35,.15,.1];options.enabled=[true,true,true,false,false]}if(name==='detail'){options.thresholds=[.09,.008,.08,.035,.28];options.weights=[1,.85,.6,.5,.3]}document.querySelectorAll('[data-style]').forEach(el=>el.classList.toggle('active',el.dataset.style===name));syncControls();schedule()}
document.querySelectorAll('[data-style]').forEach(el=>el.onclick=()=>applyStyle(el.dataset.style));
$('#save').onclick=()=>{const a=document.createElement('a');a.href=canvas.toDataURL('image/png');a.download=`lego-${options.mode}-${Date.now()}.png`;a.click()};

(async()=>{try{const r=await fetch('/api/info');if(!r.ok)throw new Error('无法连接本地渲染器');info=await r.json();options=copy(info.defaults);pose=copy(info.presets[0]);defaultPose=copy(pose);$('#count').textContent=info.gaussians.toLocaleString();$('#gpu').textContent=info.gpu;$('#camera').innerHTML=info.presets.map((p,i)=>`<option value="${i}">${p.name}</option>`).join('');buildControls();syncControls();await loadFixedPreviews();schedule()}catch(e){$('#loading').hidden=true;$('#error').hidden=false;$('#error').textContent=e.message}})();
