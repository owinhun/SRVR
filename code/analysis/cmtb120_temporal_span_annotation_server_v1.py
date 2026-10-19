#!/usr/bin/env python3
"""Local two-annotator UI for frozen CMTB-120 temporal spans."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path("/workspace/svhalluc")
MANIFEST = ROOT / "data/cmtb120_temporal_span_annotation_blind_v1.jsonl"

HTML = r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CMTB temporal spans</title><style>body{font-family:system-ui;background:#f3f5f8;color:#17202a;margin:0}main{max-width:1050px;margin:auto;padding:18px}.card{background:white;margin:12px 0;padding:16px;border-radius:12px;box-shadow:0 2px 10px #0001}.row{display:flex;gap:12px;flex-wrap:wrap}.box{flex:1;min-width:280px;border:1px solid #d7dde5;border-radius:9px;padding:12px}.current{border-left:5px solid #16827a}.narr{border-left:5px solid #c2413b}video{width:100%;max-height:520px;background:#111;border-radius:9px}button,select,input{font-size:15px;padding:8px;margin:3px}button{border:0;border-radius:7px;background:#315efb;color:white;font-weight:650}.secondary{background:#667085}.danger{background:#b42318}.span{display:inline-block;background:#eef3f8;border-radius:6px;padding:6px;margin:3px}.warn{color:#b42318;font-weight:650}.option{padding:4px 0}</style></head><body><main><h1>CMTB-120 temporal spans</h1><div class="card"><b>모델 출력과 VAD 구간은 숨겨져 있습니다.</b> 영상 증거만 보고 구간을 표시하세요. 여러 구간이면 반복해서 추가합니다.</div><div class="card"><div class="row"><b id="progress"></b><span id="saved"></span></div><h3 id="sid"></h3><video id="video" controls preload="metadata"></video><p>현재 시각: <b id="time">0.00</b>s</p></div><div class="card"><p id="question"></p><div id="options"></div><p><b>공식 current answer:</b> <span id="answer"></span></p><p><b>ASR transcript:</b> <span id="transcript"></span></p></div><div class="row"><div class="card box current"><h3>1. Current-action visual support</h3><p>공식 current action이 실제 보이는 모든 구간</p><button onclick="markStart('current')">시작점 찍기</button><button onclick="addSpan('current')">끝점 찍고 추가</button><button class="danger" onclick="clearSpans('current')">초기화</button><div id="currentSpans"></div></div><div class="card box narr"><h3>2. Narration role / visual support</h3><label>Narrated option <select id="narratedOption"><option value="">선택</option><option>A</option><option>B</option><option>C</option><option>NONE</option><option>AMBIGUOUS</option></select></label><br><label>영상 내 occurrence <select id="narratedStatus"><option value="">선택</option><option value="spans">구간 있음</option><option value="absent">영상에 없음</option><option value="ambiguous">경계 애매</option></select></label><br><button onclick="markStart('narrated')">시작점 찍기</button><button onclick="addSpan('narrated')">끝점 찍고 추가</button><button class="danger" onclick="clearSpans('narrated')">초기화</button><div id="narratedSpans"></div></div></div><div class="card row"><label>Confidence <select id="confidence"><option value="">선택</option><option value="high">high</option><option value="medium">medium</option><option value="low">low</option></select></label><label style="flex:1">Note <input id="note" style="width:80%"></label></div><div class="card row"><button onclick="save()">저장 후 다음</button><button class="secondary" onclick="move(-1)">이전</button><button class="secondary" onclick="move(1)">다음</button><span id="error" class="warn"></span></div></main><script>let state,index=0,starts={current:null,narrated:null},spans={current:[],narrated:[]};const el=id=>document.getElementById(id);async function init(){state=await(await fetch('/api/state')).json();let first=state.items.findIndex(x=>!x.annotation);index=first<0?0:first;render()}function render(){let x=state.items[index];spans={current:[],narrated:[]};starts={current:null,narrated:null};el('progress').textContent=`${state.done}/${state.items.length} 완료 · ${index+1}/${state.items.length}`;el('sid').textContent=x.sample_id;el('video').src=x.media_url;el('question').textContent=x.question;el('options').innerHTML=Object.entries(x.options).map(([k,v])=>`<div class=option><b>${k}</b> ${v}</div>`).join('');el('answer').textContent=`${x.official_current_answer}: ${x.options[x.official_current_answer]}`;el('transcript').textContent=x.asr_transcript;el('narratedOption').value='';el('narratedStatus').value='';el('confidence').value='';el('note').value='';el('saved').textContent=x.annotation?'저장됨':'';el('error').textContent='';if(x.annotation){spans.current=x.annotation.current_spans||[];spans.narrated=x.annotation.narrated_spans||[];el('narratedOption').value=x.annotation.narrated_option;el('narratedStatus').value=x.annotation.narrated_status;el('confidence').value=x.annotation.confidence;el('note').value=x.annotation.note||''}draw()}el('video').ontimeupdate=()=>el('time').textContent=el('video').currentTime.toFixed(2);function markStart(kind){starts[kind]=el('video').currentTime;el('error').textContent=`${kind} 시작 ${starts[kind].toFixed(2)}s`}function addSpan(kind){let end=el('video').currentTime,start=starts[kind];if(start===null||end<=start){el('error').textContent='시작점을 먼저 찍고 더 뒤에서 끝점을 찍으세요.';return}spans[kind].push([+start.toFixed(3),+end.toFixed(3)]);spans[kind].sort((a,b)=>a[0]-b[0]);starts[kind]=null;draw();el('error').textContent=''}function clearSpans(kind){spans[kind]=[];starts[kind]=null;draw()}function draw(){for(const k of ['current','narrated'])el(k+'Spans').innerHTML=spans[k].map((s,i)=>`<span class=span>${s[0].toFixed(2)}–${s[1].toFixed(2)} <button class=danger onclick="removeSpan('${k}',${i})">×</button></span>`).join('')}function removeSpan(k,i){spans[k].splice(i,1);draw()}function move(d){index=(index+d+state.items.length)%state.items.length;render()}async function save(){let x=state.items[index],body={sample_id:x.sample_id,current_spans:spans.current,narrated_option:el('narratedOption').value,narrated_status:el('narratedStatus').value,narrated_spans:spans.narrated,confidence:el('confidence').value,note:el('note').value};if(!body.current_spans.length||!body.narrated_option||!body.narrated_status||!body.confidence){el('error').textContent='필수 항목을 모두 채우세요.';return}if(body.narrated_status==='spans'&&!body.narrated_spans.length){el('error').textContent='구간 있음이면 narrated span을 추가하세요.';return}if(body.narrated_status!=='spans'&&body.narrated_spans.length){el('error').textContent='구간 없음/애매이면 narrated span을 비우세요.';return}let r=await fetch('/api/annotate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});if(!r.ok){el('error').textContent=await r.text();return}location.reload()}init();</script></body></html>'''


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotator-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, required=True)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def valid_spans(value: object, duration: float) -> bool:
    return isinstance(value, list) and all(
        isinstance(span, list) and len(span) == 2 and
        isinstance(span[0], (int, float)) and isinstance(span[1], (int, float)) and
        0 <= span[0] < span[1] <= duration + 0.05 for span in value
    )


def main() -> None:
    a = args()
    items = read_jsonl(MANIFEST)
    by_id = {x["sample_id"]: x for x in items}
    output = a.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, value: object, status: int = 200) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(payload))); self.end_headers(); self.wfile.write(payload)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                payload = HTML.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(payload))); self.end_headers(); self.wfile.write(payload); return
            if path == "/api/state":
                anns = {x["sample_id"]: x for x in read_jsonl(output)}
                public = [{**x, "annotation": anns.get(x["sample_id"]), "media_url": f"/media/{x['sample_id']}"} for x in items]
                self.send_json({"items": public, "done": len(anns), "annotator_id": a.annotator_id}); return
            match = re.fullmatch(r"/media/(cmtb_\d+)", path)
            if match and match.group(1) in by_id:
                media = Path(by_id[match.group(1)]["media_path"]); size = media.stat().st_size
                start, end = 0, size - 1
                if self.headers.get("Range"):
                    m = re.match(r"bytes=(\d+)-(\d*)", self.headers["Range"])
                    if m: start = int(m.group(1)); end = int(m.group(2)) if m.group(2) else end
                    self.send_response(206); self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                else: self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(media.name)[0] or "video/mp4"); self.send_header("Accept-Ranges", "bytes"); self.send_header("Content-Length", str(end-start+1)); self.end_headers()
                with media.open("rb") as f: f.seek(start); self.wfile.write(f.read(end-start+1))
                return
            self.send_error(404)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/annotate": self.send_error(404); return
            try: body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            except Exception: self.send_json({"error": "invalid json"}, 400); return
            sid = body.get("sample_id")
            if sid not in by_id: self.send_json({"error": "unknown sample"}, 400); return
            duration = float(by_id[sid]["media_duration_sec"])
            if not valid_spans(body.get("current_spans"), duration) or not body["current_spans"]: self.send_json({"error": "invalid current spans"}, 400); return
            if body.get("narrated_option") not in {"A","B","C","NONE","AMBIGUOUS"} or body.get("narrated_status") not in {"spans","absent","ambiguous"} or body.get("confidence") not in {"high","medium","low"}: self.send_json({"error": "invalid categorical field"}, 400); return
            if not valid_spans(body.get("narrated_spans"), duration): self.send_json({"error": "invalid narrated spans"}, 400); return
            if (body["narrated_status"] == "spans") != bool(body["narrated_spans"]): self.send_json({"error": "narrated status/span mismatch"}, 400); return
            record = {"sample_id": sid, "annotator_id": a.annotator_id, "current_spans": body["current_spans"], "narrated_option": body["narrated_option"], "narrated_status": body["narrated_status"], "narrated_spans": body["narrated_spans"], "confidence": body["confidence"], "note": str(body.get("note", "")), "model_outputs_seen": False}
            old = {x["sample_id"]: x for x in read_jsonl(output)}; old[sid] = record
            output.write_text("".join(json.dumps(old[k], ensure_ascii=False, sort_keys=True)+"\n" for k in sorted(old)))
            self.send_json({"ok": True})

        def log_message(self, fmt: str, *values: object) -> None:
            return

    server = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    print(f"annotator={a.annotator_id} http://127.0.0.1:{a.port} output={output}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
