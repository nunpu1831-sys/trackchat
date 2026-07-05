"""
live_mhmr4.py — 실시간 사람 + 끊김 해결(바이너리 int16 전송 + 예측 보간)
================================================================================
원인 대응(데이터 근거):
  1) 데이터 큼 → 정점을 JSON 대신 int16 바이너리로(≈5배↓, 파싱빠름) → 프레임 더 자주 도착
  2) 늦으면 멈춤(75% 정지) → 예측(extrapolation): 프레임 늦으면 속도로 계속 미끄러짐
실행(/workspace/multi-hmr, multihmr env):
  pkill -9 -f live_mhmr
  xvfb-run -a python live_mhmr4.py
"""
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"; os.environ["EGL_DEVICE_ID"] = "0"
import sys, json, asyncio, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, cv2
from PIL import Image, ImageOps
from utils import normalize_rgb, get_focalLength_from_fieldOfView, CACHE_DIR_MULTIHMR
from model import Model
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

DEVICE = "cuda"; MODEL_NAME = "multiHMR_672_S"

def load_model():
    ckpt = torch.load(os.path.join(CACHE_DIR_MULTIHMR, MODEL_NAME + '.pt'), map_location=DEVICE)
    kwargs = {k: v for k, v in vars(ckpt['args']).items()}
    kwargs['type'] = ckpt['args'].train_return_type; kwargs['img_size'] = ckpt['args'].img_size[0]
    m = Model(**kwargs).to(DEVICE); m.load_state_dict(ckpt['model_state_dict'], strict=False)
    return m.eval()

print("[mhmr4] 모델 로딩...")
MODEL = load_model()
FACES = np.asarray(MODEL.smpl_layer['neutral_10'].bm_x.faces).astype(np.int32).reshape(-1)
NV = int(FACES.max()) + 1
IMG_SIZE = MODEL.img_size
def _K(fov=60):
    K = torch.eye(3); f = get_focalLength_from_fieldOfView(fov=fov, img_size=IMG_SIZE)
    K[0,0]=f; K[1,1]=f; K[0,-1]=IMG_SIZE//2; K[1,-1]=IMG_SIZE//2
    return K.unsqueeze(0).to(DEVICE)
K_CAM = _K()
print(f"[mhmr4] 준비완료 (verts={NV}). 8000 여세요.")

_CHAIN=[0,3,6,9,12,15]; _printed=[False]
def _rod(rv):
    import cv2; return cv2.Rodrigues(np.asarray(rv,dtype=np.float64))[0]
def head_axes(h):
    try:
        rv=h['rotvec']; rv=rv.detach().cpu().numpy() if hasattr(rv,'detach') else np.asarray(rv)
        rv=rv.reshape(-1,3); R=np.eye(3)
        for i in _CHAIN:
            if i<rv.shape[0]: R=R@_rod(rv[i])
        return (R@np.array([0.,1.,0.])).astype(np.float32),(R@np.array([0.,0.,1.])).astype(np.float32)
    except Exception:
        return np.zeros(3,np.float32),np.zeros(3,np.float32)

def hand_pts(h):
    try:
        j=h['j3d']; j=j.detach().cpu().numpy() if hasattr(j,'detach') else np.asarray(j); j=j.reshape(-1,3)
        return j[21].astype(np.float32), j[19].astype(np.float32)  # 오른손목, 오른팔꿈치
    except Exception:
        return np.zeros(3,np.float32), np.zeros(3,np.float32)

_WCHAIN=[0,3,6,9,14,17,19,21]
def wrist_axes(h):
    try:
        rv=h['rotvec']; rv=rv.detach().cpu().numpy() if hasattr(rv,'detach') else np.asarray(rv); rv=rv.reshape(-1,3)
        R=np.eye(3)
        for i in _WCHAIN:
            if i<rv.shape[0]: R=R@_rod(rv[i])
        return (R@np.array([0.,1.,0.])).astype(np.float32),(R@np.array([0.,0.,1.])).astype(np.float32)
    except Exception:
        return np.zeros(3,np.float32),np.zeros(3,np.float32)

def process(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = ImageOps.pad(ImageOps.contain(Image.fromarray(rgb), (IMG_SIZE, IMG_SIZE)), size=(IMG_SIZE, IMG_SIZE))
    x = torch.from_numpy(normalize_rgb(np.asarray(img))).unsqueeze(0).to(DEVICE)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        humans = MODEL(x, is_training=False, nms_kernel_size=3, det_thresh=0.3, K=K_CAM)
    if not _printed[0] and len(humans):
        print('[keys]', list(humans[0].keys()), flush=True); _printed[0]=True
    out=[]
    for h in humans:
        v=(h['verts_smplx'] if 'verts_smplx' in h else h['v3d']).detach().cpu().numpy().astype(np.float32)
        up,fwd=head_axes(h); wr,el=hand_pts(h); wu,wf=wrist_axes(h); out.append((v,up,fwd,wr,el,wu,wf))
    return out

def encode(ppl, ms):
    buf = bytearray(); buf += struct.pack('<HH', min(int(ms),65535), len(ppl))
    for v,up,fwd,wr,el,wu,wf in ppl:
        mn = v.min(0); size = np.maximum(v.max(0)-mn, 1e-6)
        q = np.clip(np.round((v-mn)/size*65535), 0, 65535).astype('<u2')
        buf += struct.pack('<6f', float(mn[0]),float(mn[1]),float(mn[2]), float(size[0]),float(size[1]),float(size[2]))
        buf += struct.pack('<6f', float(up[0]),float(up[1]),float(up[2]), float(fwd[0]),float(fwd[1]),float(fwd[2]))
        buf += struct.pack('<6f', float(wr[0]),float(wr[1]),float(wr[2]), float(el[0]),float(el[1]),float(el[2]))
        buf += struct.pack('<6f', float(wu[0]),float(wu[1]),float(wu[2]), float(wf[0]),float(wf[1]),float(wf[2]))
        buf += q.tobytes()
    return bytes(buf)

app = FastAPI(); LOCK = asyncio.Lock()

@app.get("/")
def index(): return HTMLResponse(CLIENT_HTML)

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept(); loop = asyncio.get_event_loop(); sent = False
    try:
        async with LOCK:
            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect": break
                b = msg.get("bytes")
                if b is None: continue
                bgr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None: continue
                t0 = loop.time()
                ppl = await loop.run_in_executor(None, process, bgr)
                if not sent:
                    await websocket.send_text(json.dumps({"faces": FACES.tolist(), "nv": NV})); sent = True
                await websocket.send_bytes(encode(ppl, int((loop.time()-t0)*1000)))
    except WebSocketDisconnect: pass
    except Exception as e:
        try: await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception: pass

CLIENT_HTML = r"""<!doctype html><html lang=ko><head><meta charset=utf-8><title>TrackChat 왕관+몽둥이 11(2-3-6 앞쥠)</title>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<style>body{margin:0;background:#0e0f13;color:#eee;font-family:sans-serif;overflow:hidden}
#gl{position:fixed;inset:0}#ui{position:fixed;left:10px;top:10px;z-index:9;background:rgba(0,0,0,.6);padding:10px;border-radius:8px;font-size:13px;max-width:330px}
button{background:#5b8cff;color:#fff;border:0;padding:8px 12px;border-radius:6px;cursor:pointer;font-weight:700}
#stat{font-size:12px;margin-top:6px;color:#9effc9}.hint{font-size:11px;color:#9aa0b4;margin-top:6px;line-height:1.5}</style></head>
<body><div id=ui><button id=cap>화면 캡처 시작</button><div id=stat>대기</div>
<div class=hint>영상 창 선택 → 3D클릭 → WASD·마우스·Space/Shift·휠·ESC</div></div><canvas id=gl></canvas>
<script>
let scene,world,cam,rend,clock,faces=null,NV=0;
let yaw=0,pitch=0,keys={},locked=false,speed=2.5;
const PAL=[[0.9,0.31,0.31],[0.31,0.63,0.9],[0.78,0.7,0.9],[0.47,0.78,0.47],[0.9,0.75,0.31],[0.7,0.47,0.86],[0.4,0.8,0.8],[0.85,0.55,0.4]];
const gl=document.getElementById('gl');
// 사람 상태: prevT/lastT(초), prevV/lastV(Float32), cvel[3], disp, mesh
let ppl=[]; let NID=0;
let selected=null, crown=null, club=null; let clubMode=3,clubFlip=1,clubHud=null; const rayc=new THREE.Raycaster(), ndc=new THREE.Vector2();
function makeCrown(){const G=new THREE.Group();const mt=new THREE.MeshStandardMaterial({color:0xffcc33,metalness:0.9,roughness:0.25,emissive:0x3a2600});
 const band=new THREE.Mesh(new THREE.CylinderGeometry(1,1,0.55,24,1,true),mt);G.add(band);
 for(let i=0;i<8;i++){const sp=new THREE.Mesh(new THREE.ConeGeometry(0.26,0.75,12),mt);const a=i/8*Math.PI*2;sp.position.set(Math.cos(a),0.5,Math.sin(a));G.add(sp);}
 const jw=new THREE.Mesh(new THREE.SphereGeometry(0.28,16,16),new THREE.MeshStandardMaterial({color:0xff2244,metalness:0.3,roughness:0.4,emissive:0x330008})); jw.position.set(0,0.15,1.02);G.add(jw);
 G.visible=false;world.add(G);return G;}
function makeClub(){const G=new THREE.Group();const wood=new THREE.MeshStandardMaterial({color:0x8a5a2b,metalness:0.1,roughness:0.85}); const handle=new THREE.Mesh(new THREE.CylinderGeometry(0.12,0.16,1.3,16),wood); handle.position.y=0.65; G.add(handle); const knob=new THREE.Mesh(new THREE.SphereGeometry(0.26,16,12),wood); knob.position.y=1.4; G.add(knob); G.visible=false;world.add(G);return G;}
let sendTime=0,rttAvg=0,recvN=0,bytesN=0,dispN=0,mLast=0,lastMs=0;
function ctr(a){let x=0,y=0,z=0,n=a.length/3;for(let i=0;i<a.length;i+=3){x+=a[i];y+=a[i+1];z+=a[i+2];}return [x/n,y/n,z/n];}
function d2(p,q){const a=p[0]-q[0],b=p[1]-q[1],c=p[2]-q[2];return a*a+b*b+c*c;}
function init(){scene=new THREE.Scene();scene.background=new THREE.Color(0x0e0f13);clock=new THREE.Clock();
 world=new THREE.Group();world.scale.set(1,-1,-1);scene.add(world);
 cam=new THREE.PerspectiveCamera(60,innerWidth/innerHeight,0.01,100);cam.rotation.order='YXZ';
 cam.position.set(0,0,0.2);cam.lookAt(0,0,-3);yaw=cam.rotation.y;pitch=cam.rotation.x;
 rend=new THREE.WebGLRenderer({canvas:gl,antialias:true});rend.setSize(innerWidth,innerHeight);
 scene.add(new THREE.AmbientLight(0xffffff,0.9));const dl=new THREE.DirectionalLight(0xffffff,0.6);dl.position.set(1,-2,-1);scene.add(dl);
 addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();rend.setSize(innerWidth,innerHeight)});
 gl.addEventListener('click',(ev)=>{if(locked)return; if(!crown)crown=makeCrown(); if(!club)club=makeClub(); ndc.x=ev.clientX/innerWidth*2-1; ndc.y=-(ev.clientY/innerHeight*2)+1; rayc.setFromCamera(ndc,cam); ppl.forEach(p=>p.mesh.geometry.computeBoundingSphere()); const hit=rayc.intersectObjects(ppl.map(p=>p.mesh),false); if(hit.length){const hm=hit[0].object,p=ppl.find(q=>q.mesh===hm); if(p){selected=p; stat.textContent='왕관 대상 선택됨 · 빈 곳 클릭=시점이동';}} else gl.requestPointerLock();});
 document.addEventListener('pointerlockchange',()=>locked=document.pointerLockElement===gl);
 addEventListener('mousemove',e=>{if(!locked)return;yaw-=e.movementX*0.0025;pitch-=e.movementY*0.0025;pitch=Math.max(-1.5,Math.min(1.5,pitch));});
 addEventListener('keydown',e=>keys[e.code]=true);addEventListener('keyup',e=>keys[e.code]=false);
 addEventListener('wheel',e=>{speed=Math.max(0.3,Math.min(40,speed*(e.deltaY<0?1.15:0.87)));},{passive:true});
 clubHud=document.createElement('div'); clubHud.style.cssText='position:fixed;left:10px;bottom:10px;z-index:9;background:rgba(0,0,0,.6);color:#ffd8a0;padding:6px 10px;border-radius:6px;font-size:12px'; clubHud.textContent='몽둥이 축 mode 3(기본) · C=축바꾸기 V=뒤집기'; document.body.appendChild(clubHud);
 addEventListener('keydown',e=>{if(e.code==='KeyC'){clubMode=(clubMode+1)%4;}else if(e.code==='KeyV'){clubFlip*=-1;}else return; clubHud.textContent='몽둥이 축 mode '+clubMode+(clubFlip<0?' 뒤집음':'')+' · C=축바꾸기 V=뒤집기';});
 (function loop(){requestAnimationFrame(loop);const dt=Math.min(0.05,clock.getDelta());const now=performance.now()/1000;
  cam.rotation.x=pitch;cam.rotation.y=yaw;cam.rotation.z=0;
  const fw=new THREE.Vector3(0,0,-1).applyQuaternion(cam.quaternion),rt=new THREE.Vector3(1,0,0).applyQuaternion(cam.quaternion),vv=new THREE.Vector3();
  if(keys['KeyW'])vv.add(fw);if(keys['KeyS'])vv.sub(fw);if(keys['KeyD'])vv.add(rt);if(keys['KeyA'])vv.sub(rt);
  if(keys['Space'])vv.y+=1;if(keys['ShiftLeft']||keys['ShiftRight'])vv.y-=1;
  if(vv.lengthSq()>0){vv.normalize().multiplyScalar(speed*dt);cam.position.add(vv);}
  const DELAY=0.15, rT=now-DELAY, s=Math.min(1,dt*25);
  for(const p of ppl){
    const h=p.hist,d=p.disp; let EA,EB,a;
    if(rT<=h[0].t){EA=h[0];EB=h[0];a=0;}
    else if(rT>=h[h.length-1].t){EA=h[h.length-1];EB=EA;a=0;}
    else{let i=h.length-2;while(i>0&&h[i].t>rT)i--;EA=h[i];EB=h[i+1];a=(rT-h[i].t)/Math.max(1e-3,h[i+1].t-h[i].t);}
    const A=EA.v,B=EB.v;
    for(let k=0;k<d.length;k++){const tk=A[k]+(B[k]-A[k])*a; d[k]+=(tk-d[k])*s;}
    if(EA.w&&EB.w){if(!p.dw){p.dw=EA.w.slice();p.de=EA.e.slice();p.dwu=EA.wu.slice();p.dwf=EA.wf.slice();} for(let k=0;k<3;k++){p.dw[k]+=((EA.w[k]+(EB.w[k]-EA.w[k])*a)-p.dw[k])*s; p.de[k]+=((EA.e[k]+(EB.e[k]-EA.e[k])*a)-p.de[k])*s; p.dwu[k]+=((EA.wu[k]+(EB.wu[k]-EA.wu[k])*a)-p.dwu[k])*s; p.dwf[k]+=((EA.wf[k]+(EB.wf[k]-EA.wf[k])*a)-p.dwf[k])*s;}}
    const pos=p.mesh.geometry.attributes.position; pos.array.set(d); pos.needsUpdate=true; p.mesh.geometry.computeVertexNormals();
  }
  dispN++;
  if(now-mLast>0.5){const el=Math.max(0.001,now-mLast);
    const rf=recvN/el,df=dispN/el,kbf=recvN?bytesN/recvN/1024:0,kbps=bytesN/el/1024,net=Math.max(0,rttAvg-lastMs);
    stat.innerHTML='<b>받은 '+rf.toFixed(0)+'fps</b> · 화면 '+df.toFixed(0)+'fps · 사람 '+ppl.length+'<br>RTT '+rttAvg.toFixed(0)+'ms = 추론 '+lastMs+'ms + 네트워크 '+net.toFixed(0)+'ms<br>프레임당 '+kbf.toFixed(0)+'KB · 대역폭 '+kbps.toFixed(0)+'KB/s';
    recvN=0;bytesN=0;dispN=0;mLast=now;}
  if(crown){const p=selected&&ppl.indexOf(selected)>=0?selected:null; if(p){const d=p.disp; let miY=1e9,maY=-1e9; for(let k=1;k<d.length;k+=3){const y=d[k];if(y<miY)miY=y;if(y>maY)maY=y;} const H=Math.max(1e-3,maY-miY),topT=miY+0.05*H,midT=miY+0.20*H; let tx=0,ty=0,tz=0,tn=0,mx=0,my=0,mz=0,mn=0; for(let k=0;k<d.length;k+=3){const y=d[k+1]; if(y<=topT){tx+=d[k];ty+=y;tz+=d[k+2];tn++;} if(y<=midT){mx+=d[k];my+=y;mz+=d[k+2];mn++;}} tx/=tn;ty/=tn;tz/=tn;mx/=mn;my/=mn;mz/=mn; let ux,uy,uz,fx,fy,fz; const o=p.ori; if(o&&(o.up[0]||o.up[1]||o.up[2])){ux=o.up[0];uy=o.up[1];uz=o.up[2];fx=o.fwd[0];fy=o.fwd[1];fz=o.fwd[2];} else {ux=tx-mx;uy=ty-my;uz=tz-mz;fx=0;fy=0;fz=1;} let ul=Math.hypot(ux,uy,uz)||1; ux/=ul;uy/=ul;uz/=ul; let fd=fx*ux+fy*uy+fz*uz; fx-=fd*ux;fy-=fd*uy;fz-=fd*uz; let fl=Math.hypot(fx,fy,fz); if(fl<1e-4){fx=1-ux*ux;fy=-ux*uy;fz=-ux*uz;fl=Math.hypot(fx,fy,fz)||1;} fx/=fl;fy/=fl;fz/=fl; let rx=uy*fz-uz*fy, ry=uz*fx-ux*fz, rz=ux*fy-uy*fx; let wsum=0,wn=0; for(let k=0;k<d.length;k+=3){const y=d[k+1]; if(y<=midT){const ax=d[k]-mx,ay=y-my,az=d[k+2]-mz,dt2=ax*ux+ay*uy+az*uz,ex=ax-dt2*ux,ey=ay-dt2*uy,ez=az-dt2*uz; wsum+=Math.hypot(ex,ey,ez); wn++;}} const ts=Math.max(1e-3,(wsum/Math.max(1,wn))*1.35); const M=new THREE.Matrix4(); M.makeBasis(new THREE.Vector3(rx,ry,rz),new THREE.Vector3(ux,uy,uz),new THREE.Vector3(fx,fy,fz)); const tq=new THREE.Quaternion().setFromRotationMatrix(M); const tp=new THREE.Vector3(tx+ux*ts*0.35,ty+uy*ts*0.35,tz+uz*ts*0.35); if(p.cP===undefined){p.cP=tp.clone();p.cQ=tq.clone();p.cS=ts;} const smP=Math.min(1,dt*10), smS=Math.min(1,dt*4); p.cP.lerp(tp,smP); p.cQ.slerp(tq,smP); p.cS+=(ts-p.cS)*smS; crown.visible=true; crown.position.copy(p.cP); crown.quaternion.copy(p.cQ); crown.scale.set(p.cS,p.cS,p.cS);} else crown.visible=false;}
  if(club){const p=(selected&&ppl.indexOf(selected)>=0&&selected.dw&&(Math.abs(selected.dw[0])+Math.abs(selected.dw[1])+Math.abs(selected.dw[2])>1e-4))?selected:null; if(p){const w=p.dw,e=p.de; let f0x=w[0]-e[0],f0y=w[1]-e[1],f0z=w[2]-e[2]; let L=Math.hypot(f0x,f0y,f0z)||1e-3; f0x/=L;f0y/=L;f0z/=L; const scl=Math.max(0.05,L*1.1); let dx=f0x,dy=f0y,dz=f0z; const u=p.dwu,fw=p.dwf; if(u&&fw&&(u[0]||u[1]||u[2])){ const rx=u[1]*fw[2]-u[2]*fw[1], ry=u[2]*fw[0]-u[0]*fw[2], rz=u[0]*fw[1]-u[1]*fw[0]; let ax,ay,az; if(clubMode===0){ax=f0x;ay=f0y;az=f0z;} else if(clubMode===1){ax=u[0];ay=u[1];az=u[2];} else if(clubMode===2){ax=fw[0];ay=fw[1];az=fw[2];} else {ax=rx;ay=ry;az=rz;} const al=Math.hypot(ax,ay,az)||1; ax/=al;ay/=al;az/=al; let sg=clubFlip; if(clubMode===3){ sg=clubFlip*(((ax*f0x+ay*f0y+az*f0z)>=0)?1:-1); } dx=sg*ax;dy=sg*ay;dz=sg*az; } const gp=scl*0.30; const tp=new THREE.Vector3(w[0]+dx*gp,w[1]+dy*gp,w[2]+dz*gp); const tq=new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0,1,0),new THREE.Vector3(dx,dy,dz)); if(p.kP===undefined){p.kP=tp.clone();p.kQ=tq.clone();p.kS=scl;} const sm=Math.min(1,dt*12); p.kP.lerp(tp,sm); p.kQ.slerp(tq,sm); p.kS+=(scl-p.kS)*Math.min(1,dt*6); club.visible=true; club.position.copy(p.kP); club.quaternion.copy(p.kQ); club.scale.set(p.kS,p.kS,p.kS); } else club.visible=false;}
  rend.render(scene,cam);})();}
function makeMesh(i,v){const g=new THREE.BufferGeometry();
 g.setAttribute('position',new THREE.BufferAttribute(v.slice(),3));g.setIndex(new THREE.BufferAttribute(new Uint32Array(faces),1));g.computeVertexNormals();
 const c=PAL[i%PAL.length];const m=new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:new THREE.Color(c[0],c[1],c[2]),side:THREE.DoubleSide}));world.add(m);return m;}
function onFrame(list,oris){const now=performance.now()/1000;
 const newC=list.map(ctr);
 const oldC=ppl.map(p=>ctr(p.hist[p.hist.length-1].v));
 const usedP=new Array(ppl.length).fill(false), md=new Array(list.length).fill(false);
 for(let i=0;i<list.length;i++){let best=-1,bd=0.35;
   for(let j=0;j<ppl.length;j++){if(usedP[j])continue;const dd=d2(newC[i],oldC[j]);if(dd<bd){bd=dd;best=j;}}
   if(best>=0){usedP[best]=true;md[i]=true;const p=ppl[best];p.hist.push({t:now,v:list[i],w:oris[i].w,e:oris[i].e,wu:oris[i].wu,wf:oris[i].wf});while(p.hist.length>8)p.hist.shift();p.ori=oris[i];p.seen=now;}}
 for(let i=0;i<list.length;i++){if(!md[i]){const p={id:NID++,hist:[{t:now,v:list[i],w:oris[i].w,e:oris[i].e,wu:oris[i].wu,wf:oris[i].wf}],disp:list[i].slice(),dw:oris[i].w.slice(),de:oris[i].e.slice(),dwu:oris[i].wu.slice(),dwf:oris[i].wf.slice(),ori:oris[i],seen:now};p.mesh=makeMesh(p.id,list[i]);ppl.push(p);}}
 for(let j=ppl.length-1;j>=0;j--){if(now-ppl[j].seen>1.2){if(selected===ppl[j])selected=null;world.remove(ppl[j].mesh);ppl.splice(j,1);}}
}
const stat=document.getElementById('stat');
init();
const cvs=document.createElement('canvas'),cx=cvs.getContext('2d');let ws,inflight=0,pv=null;
document.getElementById('cap').onclick=async()=>{
 const st=await navigator.mediaDevices.getDisplayMedia({video:{frameRate:30},audio:false});
 pv=document.createElement('video');pv.srcObject=st;await pv.play();
 const proto=location.protocol==='https:'?'wss':'ws';ws=new WebSocket(proto+'://'+location.host+'/ws');ws.binaryType='arraybuffer';
 ws.onopen=()=>{stat.textContent='캡처+연결됨';setInterval(send,33);};
 ws.onmessage=e=>{inflight=0;
   if(typeof e.data==='string'){const d=JSON.parse(e.data);if(d.error){stat.textContent='에러:'+d.error;return;}if(d.faces){faces=d.faces;NV=d.nv;}return;}
   if(!faces)return;const buf=e.data,dv=new DataView(buf);const ms=dv.getUint16(0,true),P=dv.getUint16(2,true);let off=4;const list=[],oris=[];
   for(let k=0;k<P;k++){
     const mnx=dv.getFloat32(off,true),mny=dv.getFloat32(off+4,true),mnz=dv.getFloat32(off+8,true);
     const sx=dv.getFloat32(off+12,true),sy=dv.getFloat32(off+16,true),sz=dv.getFloat32(off+20,true);off+=24; const ux=dv.getFloat32(off,true),uy=dv.getFloat32(off+4,true),uz=dv.getFloat32(off+8,true),fx=dv.getFloat32(off+12,true),fy=dv.getFloat32(off+16,true),fz=dv.getFloat32(off+20,true);off+=24; const wx=dv.getFloat32(off,true),wy=dv.getFloat32(off+4,true),wz=dv.getFloat32(off+8,true),elx=dv.getFloat32(off+12,true),ely=dv.getFloat32(off+16,true),elz=dv.getFloat32(off+20,true);off+=24; const wux=dv.getFloat32(off,true),wuy=dv.getFloat32(off+4,true),wuz=dv.getFloat32(off+8,true),wfx=dv.getFloat32(off+12,true),wfy=dv.getFloat32(off+16,true),wfz=dv.getFloat32(off+20,true);off+=24; oris.push({up:[ux,uy,uz],fwd:[fx,fy,fz],w:[wx,wy,wz],e:[elx,ely,elz],wu:[wux,wuy,wuz],wf:[wfx,wfy,wfz]});
     const q=new Uint16Array(buf,off,NV*3);off+=NV*3*2;const v=new Float32Array(NV*3);
     for(let i=0;i<NV;i++){v[i*3]=mnx+q[i*3]/65535*sx;v[i*3+1]=mny+q[i*3+1]/65535*sy;v[i*3+2]=mnz+q[i*3+2]/65535*sz;}
     list.push(v);}
   const rtt=performance.now()-sendTime;rttAvg=rttAvg?rttAvg*0.8+rtt*0.2:rtt;recvN++;bytesN+=buf.byteLength;lastMs=ms;
   onFrame(list,oris);};
 ws.onclose=()=>stat.textContent='연결 종료';};
function send(){if(!ws||ws.readyState!==1||inflight||!pv||!pv.videoWidth)return;inflight=1;sendTime=performance.now();
 cvs.width=640;cvs.height=Math.round(640*pv.videoHeight/pv.videoWidth);cx.drawImage(pv,0,0,cvs.width,cvs.height);
 cvs.toBlob(b=>{if(b&&ws.readyState===1)b.arrayBuffer().then(a=>ws.send(a));else inflight=0;},'image/jpeg',0.6);}
</script></body></html>"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
