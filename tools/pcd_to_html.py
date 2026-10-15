#!/usr/bin/env python3
"""Downsample a binary PCD (x y z intensity) and emit a self-contained WebGL
point-cloud viewer HTML (no external libs, no CDN — CSP-safe for Artifacts).
Positions -> Int16 (single scale), colours -> Uint8 (turbo by height)."""
import os, sys, struct, base64, numpy as np

if len(sys.argv) < 2:
    sys.exit('usage: pcd_to_html.py <map.pcd> [out.html] [name]')
PCD = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else PCD.rsplit('.', 1)[0] + '_viewer.html'
NAME = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(PCD).rsplit('.', 1)[0]
TARGET = 350_000          # ~ target point count after voxel downsample

# ---- read PCD header + binary body ----
with open(PCD, 'rb') as f:
    raw = f.read()
hdr_end = raw.index(b'DATA binary\n') + len(b'DATA binary\n')
header = raw[:hdr_end].decode('ascii', 'replace')
npts = int([l for l in header.splitlines() if l.startswith('POINTS')][0].split()[1])
body = raw[hdr_end:]
pts = np.frombuffer(body, dtype=np.float32, count=npts * 4).reshape(-1, 4)
xyz = pts[:, :3].astype(np.float64)
xyz = xyz[np.isfinite(xyz).all(1)]
print(f'loaded {len(xyz):,} finite pts')

# ---- voxel downsample: keep one point per voxel, size chosen to hit TARGET ----
mn = xyz.min(0); ext = xyz.max(0) - mn
vox = max(ext) / 400.0
for _ in range(24):
    keys = np.floor((xyz - mn) / vox).astype(np.int64)
    k1 = (keys[:, 0] << 42) ^ (keys[:, 1] << 21) ^ keys[:, 2]
    uniq = np.unique(k1, return_index=True)[1]
    if len(uniq) <= TARGET * 1.15:
        break
    vox *= 1.15
ds = xyz[uniq]
print(f'downsampled -> {len(ds):,} pts (voxel {vox*1000:.0f} mm)')

# ---- centre + Int16 quantise ----
c = ds.mean(0)
p = ds - c
scale = 32000.0 / np.abs(p).max()
q = np.round(p * scale).astype(np.int16)          # (N,3) int16

# ---- colour by height (turbo) ----
z = ds[:, 2]
lo, hi = np.percentile(z, 1), np.percentile(z, 99)
t = np.clip((z - lo) / max(1e-6, hi - lo), 0, 1)
try:
    import matplotlib.cm as cm
    rgb = (cm.get_cmap('turbo')(t)[:, :3] * 255).astype(np.uint8)
except Exception:                                  # fallback ramp blue->red
    rgb = np.stack([t, np.sin(t*np.pi), 1 - t], 1)
    rgb = (rgb * 255).astype(np.uint8)

# ---- pack: int16 xyz | uint8 rgb, base64 ----
buf = q.tobytes() + np.ascontiguousarray(rgb).tobytes()
b64 = base64.b64encode(buf).decode()
N = len(ds)
meta = dict(n=N, invScale=float(1.0 / scale),
            zlo=float(lo - c[2]), zhi=float(hi - c[2]))
print(f'packed {len(buf)/1e6:.1f} MB -> base64 {len(b64)/1e6:.1f} MB')

HTML = f"""<div id="wrap">
  <canvas id="gl"></canvas>
  <div id="hud">
    <div class="card">
      <div class="eyebrow">GLIM point cloud</div>
      <h1>{NAME}</h1>
      <div class="stat"><span>{N:,}</span> points shown&nbsp;·&nbsp;<span>{npts:,}</span> total</div>
    </div>
  </div>
  <div id="legend">
    <span class="lz">{meta['zhi']:.2f} m</span>
    <div class="bar"></div>
    <span class="lz">{meta['zlo']:.2f} m</span>
    <div class="cap">height</div>
  </div>
  <div id="help">drag&nbsp;rotate&nbsp;·&nbsp;scroll&nbsp;zoom&nbsp;·&nbsp;shift-drag&nbsp;pan&nbsp;·&nbsp;<b id="pt">2px</b></div>
</div>
<style>
  :root {{
    --bg:#0a0d13; --panel:rgba(18,23,33,.72); --line:rgba(120,140,170,.18);
    --ink:#cdd8e8; --dim:#7f8da3; --accent:#54d6c4;
  }}
  * {{ box-sizing:border-box; }}
  #wrap {{ position:fixed; inset:0; background:
      radial-gradient(120% 120% at 50% 0%, #121826 0%, var(--bg) 60%);
      overflow:hidden; font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }}
  #gl {{ position:absolute; inset:0; width:100%; height:100%; display:block; cursor:grab; }}
  #gl:active {{ cursor:grabbing; }}
  #hud {{ position:absolute; top:18px; left:18px; }}
  .card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px;
      padding:14px 18px; backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); }}
  .eyebrow {{ font:600 10px/1 ui-monospace,SFMono-Regular,Menlo,monospace;
      letter-spacing:.18em; text-transform:uppercase; color:var(--accent); }}
  h1 {{ margin:6px 0 8px; font-size:22px; font-weight:650; color:var(--ink);
      letter-spacing:-.01em; }}
  .stat {{ font:500 12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace; color:#9fb0c9;
      font-variant-numeric:tabular-nums; }}
  .stat span {{ color:var(--ink); }}
  #legend {{ position:absolute; top:18px; right:18px; display:flex; flex-direction:column;
      align-items:center; gap:6px; background:var(--panel); border:1px solid var(--line);
      border-radius:12px; padding:12px 12px 10px; backdrop-filter:blur(10px);
      -webkit-backdrop-filter:blur(10px); }}
  .bar {{ width:12px; height:160px; border-radius:6px; border:1px solid var(--line);
      background:linear-gradient(to top,
        #30123b,#4145ab,#2ca4b7,#3ee273,#c3f333,#fdad33,#e5460b,#900c00); }}
  .lz {{ font:500 10px/1 ui-monospace,Menlo,monospace; color:#9fb0c9;
      font-variant-numeric:tabular-nums; }}
  .cap {{ font:600 9px/1 ui-monospace,Menlo,monospace; letter-spacing:.16em;
      text-transform:uppercase; color:var(--dim); margin-top:2px; }}
  #help {{ position:absolute; bottom:16px; left:50%; transform:translateX(-50%);
      font:500 11px/1 ui-monospace,Menlo,monospace; color:#8a9ab4; letter-spacing:.02em;
      background:var(--panel); border:1px solid var(--line); border-radius:999px;
      padding:8px 16px; backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px);
      user-select:none; }}
  #help b {{ color:var(--accent); }}
</style>
<script>
const META = {{n:{N}, invScale:{meta['invScale']:.9g}}};
const B64 = "{b64}";
function decode(b64) {{
  const bin = atob(b64), len = bin.length, u8 = new Uint8Array(len);
  for (let i=0;i<len;i++) u8[i]=bin.charCodeAt(i);
  return u8.buffer;
}}
const buf = decode(B64), N = META.n;
const pos = new Int16Array(buf, 0, N*3);
const col = new Uint8Array(buf, N*3*2, N*3);

const cv = document.getElementById('gl');
const gl = cv.getContext('webgl', {{antialias:true, alpha:false}});
const vs = `attribute vec3 aPos; attribute vec3 aCol; uniform mat4 uMVP;
  uniform float uSize; varying vec3 vc;
  void main(){{ gl_Position = uMVP * vec4(aPos*{meta['invScale']:.9g}, 1.0);
    gl_PointSize = uSize; vc = aCol; }}`;
const fs = `precision mediump float; varying vec3 vc;
  void main(){{ vec2 d = gl_PointCoord - 0.5; if (dot(d,d) > 0.25) discard;
    gl_FragColor = vec4(vc, 1.0); }}`;
function sh(t,s){{ const o=gl.createShader(t); gl.shaderSource(o,s); gl.compileShader(o);
  if(!gl.getShaderParameter(o,gl.COMPILE_STATUS)) throw gl.getShaderInfoLog(o); return o; }}
const prog = gl.createProgram();
gl.attachShader(prog, sh(gl.VERTEX_SHADER, vs));
gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fs));
gl.linkProgram(prog); gl.useProgram(prog);
const pb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, pb);
gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
const aPos = gl.getAttribLocation(prog,'aPos');
gl.enableVertexAttribArray(aPos); gl.vertexAttribPointer(aPos,3,gl.SHORT,false,0,0);
const cb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, cb);
gl.bufferData(gl.ARRAY_BUFFER, col, gl.STATIC_DRAW);
const aCol = gl.getAttribLocation(prog,'aCol');
gl.enableVertexAttribArray(aCol); gl.vertexAttribPointer(aCol,3,gl.UNSIGNED_BYTE,true,0,0);
const uMVP = gl.getUniformLocation(prog,'uMVP'), uSize = gl.getUniformLocation(prog,'uSize');
gl.clearColor(0.039,0.051,0.075,1); gl.enable(gl.DEPTH_TEST);

// ---- camera (z-up orbit) ----
let yaw=0.7, pitch=0.9, dist=2.4, tgt=[0,0,0], size=2.0;
function mul(a,b){{const o=new Float32Array(16);for(let i=0;i<4;i++)for(let j=0;j<4;j++){{
  let s=0;for(let k=0;k<4;k++)s+=a[k*4+j]*b[i*4+k];o[i*4+j]=s;}}return o;}}
function persp(f,a,n,fa){{const t=1/Math.tan(f/2);return new Float32Array(
  [t/a,0,0,0, 0,t,0,0, 0,0,(fa+n)/(n-fa),-1, 0,0,2*fa*n/(n-fa),0]);}}
function look(e,c,u){{const z=[e[0]-c[0],e[1]-c[1],e[2]-c[2]];let l=Math.hypot(...z);
  z[0]/=l;z[1]/=l;z[2]/=l;
  let x=[u[1]*z[2]-u[2]*z[1],u[2]*z[0]-u[0]*z[2],u[0]*z[1]-u[1]*z[0]];
  l=Math.hypot(...x)||1;x=x.map(v=>v/l);
  const y=[z[1]*x[2]-z[2]*x[1],z[2]*x[0]-z[0]*x[2],z[0]*x[1]-z[1]*x[0]];
  return new Float32Array([x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0,
    -(x[0]*e[0]+x[1]*e[1]+x[2]*e[2]), -(y[0]*e[0]+y[1]*e[1]+y[2]*e[2]),
    -(z[0]*e[0]+z[1]*e[1]+z[2]*e[2]), 1]);}}
function draw(){{
  const w=cv.clientWidth, h=cv.clientHeight, dpr=Math.min(2,devicePixelRatio||1);
  if(cv.width!==w*dpr||cv.height!==h*dpr){{cv.width=w*dpr;cv.height=h*dpr;}}
  gl.viewport(0,0,cv.width,cv.height);
  gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  const cp=Math.cos(pitch),e=[tgt[0]+dist*cp*Math.cos(yaw), tgt[1]+dist*cp*Math.sin(yaw),
    tgt[2]+dist*Math.sin(pitch)];
  const mvp=mul(persp(1.05,w/h,0.01,50), look(e,tgt,[0,0,1]));
  gl.uniformMatrix4fv(uMVP,false,mvp); gl.uniform1f(uSize,size*dpr);
  gl.drawArrays(gl.POINTS,0,N);
}}
function loop(){{ draw(); requestAnimationFrame(loop); }} loop();

// ---- interaction ----
let drag=null;
cv.addEventListener('mousedown',e=>{{drag={{x:e.clientX,y:e.clientY,pan:e.shiftKey||e.button===2}};}});
addEventListener('mouseup',()=>drag=null);
addEventListener('mousemove',e=>{{ if(!drag)return;
  const dx=e.clientX-drag.x, dy=e.clientY-drag.y; drag.x=e.clientX; drag.y=e.clientY;
  if(drag.pan){{ const cp=Math.cos(pitch);
    const rx=[-Math.sin(yaw),Math.cos(yaw),0];
    const uz=[-Math.sin(pitch)*Math.cos(yaw),-Math.sin(pitch)*Math.sin(yaw),Math.cos(pitch)];
    const s=dist*0.0015;
    for(let i=0;i<3;i++) tgt[i]+=(-dx*rx[i]+dy*uz[i])*s;
  }} else {{ yaw-=dx*0.006; pitch=Math.max(-1.5,Math.min(1.5,pitch+dy*0.006)); }}
}});
cv.addEventListener('contextmenu',e=>e.preventDefault());
cv.addEventListener('wheel',e=>{{ e.preventDefault();
  dist*=Math.exp(e.deltaY*0.001); dist=Math.max(0.15,Math.min(20,dist)); }},{{passive:false}});
addEventListener('keydown',e=>{{ if(e.key==='['||e.key===']'){{
  size=Math.max(1,Math.min(6,size+(e.key===']'?0.5:-0.5)));
  document.getElementById('pt').textContent=size+'px'; }} }});
</script>
"""

with open(OUT, 'w') as f:
    f.write(HTML)
print('wrote', OUT, f'({len(HTML)/1e6:.1f} MB)')
