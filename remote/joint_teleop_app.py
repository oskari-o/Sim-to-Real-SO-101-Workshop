#!/usr/bin/env python3
"""Joint-space teleop for the SO-101 sim, without a leader arm.

Run locally (Mac). Standard library only, no venv needed:
    python3 remote/joint_teleop_app.py
then open http://127.0.0.1:8765 and drag sliders / use the keyboard.

The app listens for the simulator on 127.0.0.1:54322. Forward that port to the
GPU server so lerobot_agent.py (running with --teleop_device virtual) can reach it:
    ssh -N -R 54322:localhost:54322 upcloud_gpu

Values are in LeRobot leader units, exactly what a physical SO101 leader reports:
arm joints -100..100, gripper 0..100. Recorded datasets look the same as leader-arm ones.
See source/sim_to_real_so101/utils/virtual_leader.py for the wire protocol.
"""
from __future__ import annotations

import argparse
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
          "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]


class SimLink:
    """TCP server the simulator's VirtualLeader connects to."""

    def __init__(self, port: int):
        self.lock = threading.Lock()
        self.conn: socket.socket | None = None
        self.state = {j: 0.0 for j in JOINTS}
        self.home = dict(self.state)
        self.state_seq = 0  # bumps whenever the sim pushes a pose the sliders must adopt
        self.recording = False
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", port))
        srv.listen(1)
        self.srv = srv
        threading.Thread(target=self._accept_loop, daemon=True).start()

    @property
    def connected(self) -> bool:
        return self.conn is not None

    def send(self, msg: dict):
        with self.lock:
            c = self.conn
        if c is None:
            return False
        try:
            c.sendall((json.dumps(msg) + "\n").encode())
            return True
        except OSError:
            return False

    def _accept_loop(self):
        while True:
            c, _ = self.srv.accept()
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self.lock:
                if self.conn:
                    self.conn.close()
                self.conn = c
            print("[app] simulator connected")
            threading.Thread(target=self._read_loop, args=(c,), daemon=True).start()

    def _read_loop(self, c: socket.socket):
        buf = b""
        try:
            while True:
                chunk = c.recv(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    msg = json.loads(line)
                    with self.lock:
                        if msg.get("type") == "state":
                            self.state = {j: float(msg["action"][j]) for j in JOINTS}
                            if "home" in msg:
                                self.home = {j: float(msg["home"][j]) for j in JOINTS}
                            self.state_seq += 1
                        elif msg.get("type") == "status":
                            self.recording = bool(msg.get("recording"))
        except (OSError, json.JSONDecodeError):
            pass
        finally:
            with self.lock:
                if self.conn is c:
                    self.conn = None
                    self.recording = False
            c.close()
            print("[app] simulator disconnected")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SO-101 Joint Teleop</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e2dd;--acc:#e8702a;--ok:#2f9e57;--bad:#c94a3c;--rec:#d23b3b}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#20201e;--fg:#ecece8;--mut:#9a9a93;--line:#34342f;--acc:#f08a45}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 -apple-system,system-ui,sans-serif}
main{max-width:760px;margin:0 auto;padding:20px 16px 40px}
header{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:14px}
h1{font-size:18px;margin:0;font-weight:650}
.pill{font-size:12px;padding:3px 10px;border-radius:99px;border:1px solid var(--line);color:var(--mut)}
.pill.ok{color:var(--ok);border-color:var(--ok)}.pill.rec{color:#fff;background:var(--rec);border-color:var(--rec)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:6px 16px}
.row{display:grid;grid-template-columns:120px 1fr 96px;gap:12px;align-items:center;padding:10px 0;border-bottom:1px solid var(--line)}
.row:last-child{border-bottom:0}.name{font-weight:550}.keys{font-size:11px;color:var(--mut);font-family:ui-monospace,monospace}
input[type=range]{width:100%;accent-color:var(--acc)}
.val{text-align:right;font-variant-numeric:tabular-nums}.deg{display:block;font-size:11px;color:var(--mut)}
.bar{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}
button{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button:hover{border-color:var(--acc)}button.primary{background:var(--acc);border-color:var(--acc);color:#fff}
button.rec{background:var(--rec);border-color:var(--rec);color:#fff}
.speed{display:flex;align-items:center;gap:8px;color:var(--mut);font-size:13px;margin-left:auto}
.help{color:var(--mut);font-size:13px;margin-top:14px}
kbd{font-family:ui-monospace,monospace;font-size:12px;border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px}
.dis .card{opacity:.55}
@media (max-width:520px){.row{grid-template-columns:1fr 80px}.row input{grid-column:1/3;grid-row:2}}
</style></head>
<body><main id="root" class="dis">
<header><h1>SO-101 joint teleop</h1><span id="conn" class="pill">waiting for sim</span><span id="rec" class="pill" hidden>REC</span></header>
<div class="card" id="joints"></div>
<div class="bar">
  <button id="b-rec" class="primary">Start recording <kbd>Space</kbd></button>
  <button id="b-cancel">Cancel episode</button>
  <button id="b-home">Home</button>
  <button id="b-reset">Reset world</button>
  <label class="speed">speed <input id="speed" type="range" min="10" max="200" value="60"> <span id="speedv">60</span>/s</label>
</div>
<p class="help">Hold the keys to move a joint (<kbd>Shift</kbd> = fine). Sliders jump straight to a value; the sim side
rate-limits so the arm won't snap. Reset puts the world and sliders back to the start pose. Recording follows the
same rules as the S / C / R keys in lerobot_agent.</p>
</main>
<script>
const J=[["shoulder_pan.pos","Shoulder pan","q","a",-100,100,-110,110],
 ["shoulder_lift.pos","Shoulder lift","w","s",-100,100,-100,100],
 ["elbow_flex.pos","Elbow","e","d",-100,100,-100,90],
 ["wrist_flex.pos","Wrist flex","r","f",-100,100,-95,95],
 ["wrist_roll.pos","Wrist roll","t","g",-100,100,-160,160],
 ["gripper.pos","Gripper","y","h",0,100,-10,100]];
const target={},el={};let seq=-1,home={},dirty=false,connected=false,recording=false;
const box=document.getElementById("joints");
for(const [k,n,up,dn,lo,hi] of J){
  const r=document.createElement("div");r.className="row";
  r.innerHTML=`<div><div class="name">${n}</div><div class="keys">${up.toUpperCase()} + / ${dn.toUpperCase()} −</div></div>
  <input type="range" min="${lo}" max="${hi}" step="0.1" value="0"><div class="val"><span>0.0</span><span class="deg"></span></div>`;
  box.appendChild(r);const s=r.querySelector("input");
  el[k]={s,v:r.querySelector(".val span"),d:r.querySelector(".deg")};target[k]=0;
  s.addEventListener("input",()=>{target[k]=+s.value;dirty=true;show(k)});
}
function deg(k,v){const j=J.find(x=>x[0]==k);const n=k=="gripper.pos"?v/100:(v+100)/200;return j[6]+n*(j[7]-j[6])}
function show(k){el[k].v.textContent=target[k].toFixed(1);el[k].d.textContent=deg(k,target[k]).toFixed(0)+"°";el[k].s.value=target[k]}
function setAll(p){for(const [k] of J){target[k]=p[k];show(k)}}
const post=(u,b)=>fetch(u,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b)}).catch(()=>{});
document.getElementById("b-rec").onclick=()=>post("/cmd",{cmd:"record"});
document.getElementById("b-cancel").onclick=()=>post("/cmd",{cmd:"cancel"});
document.getElementById("b-reset").onclick=()=>post("/cmd",{cmd:"reset"});
document.getElementById("b-home").onclick=()=>{setAll(home);dirty=true};
const sp=document.getElementById("speed"),spv=document.getElementById("speedv");sp.oninput=()=>spv.textContent=sp.value;
const held=new Set();
addEventListener("keydown",e=>{if(e.target.tagName=="INPUT"&&e.target.type!="range")return;
  if(e.code=="Space"){e.preventDefault();if(!e.repeat)post("/cmd",{cmd:"record"});return}
  held.add(e.key.toLowerCase())});
addEventListener("keyup",e=>held.delete(e.key.toLowerCase()));
addEventListener("blur",()=>held.clear());
let last=performance.now(),lastSend=0;
function tick(t){const dt=Math.min((t-last)/1000,0.1);last=t;
  const rate=+sp.value*(held.has("shift")?0.25:1);
  for(const [k,,up,dn,lo,hi] of J){let d=0;if(held.has(up))d+=1;if(held.has(dn))d-=1;
    if(d){target[k]=Math.max(lo,Math.min(hi,target[k]+d*rate*dt));show(k);dirty=true}}
  if(dirty&&connected&&t-lastSend>20){dirty=false;lastSend=t;post("/target",{action:target})}
  requestAnimationFrame(tick)}
requestAnimationFrame(tick);
async function poll(){try{const s=await (await fetch("/status")).json();
  connected=s.connected;home=s.home;
  document.getElementById("root").classList.toggle("dis",!connected);
  const c=document.getElementById("conn");c.textContent=connected?"sim connected":"waiting for sim";c.classList.toggle("ok",connected);
  if(connected&&s.state_seq!=seq){seq=s.state_seq;setAll(s.state)}
  if(s.recording!=recording){recording=s.recording;document.getElementById("rec").hidden=!recording;
    const b=document.getElementById("b-rec");b.className=recording?"rec":"primary";
    b.innerHTML=(recording?"Stop &amp; save":"Start recording")+" <kbd>Space</kbd>"}
 }catch(e){}setTimeout(poll,250)}
poll();
</script></body></html>"""


def make_handler(link: SimLink):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/status":
                with link.lock:
                    self._json({"connected": link.connected, "state": link.state, "home": link.home,
                                "state_seq": link.state_seq, "recording": link.recording})
            else:
                self.send_error(404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"ok": False}, 400)
            if self.path == "/target":
                act = {j: float(data["action"][j]) for j in JOINTS if j in data.get("action", {})}
                return self._json({"ok": link.send({"type": "target", "action": act})})
            if self.path == "/cmd" and data.get("cmd") in ("reset", "record", "cancel"):
                return self._json({"ok": link.send({"type": "cmd", "cmd": data["cmd"]})})
            self.send_error(404)

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-port", type=int, default=54322, help="TCP port the simulator connects to")
    ap.add_argument("--http-port", type=int, default=8765, help="port for the browser UI")
    a = ap.parse_args()
    link = SimLink(a.sim_port)
    http = ThreadingHTTPServer(("127.0.0.1", a.http_port), make_handler(link))
    print(f"[app] UI on http://127.0.0.1:{a.http_port}  |  waiting for the sim on 127.0.0.1:{a.sim_port}")
    print(f"[app] tunnel it with: ssh -N -R {a.sim_port}:localhost:{a.sim_port} upcloud_gpu")
    try:
        http.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
