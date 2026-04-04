"""Local web viewer for corpus review.

Serves HTML viewer + API for random records from parquet files.
Single user, runs locally.

Usage:
    uv run --extra hf python scripts/serve_viewer.py --output-dir ../hf_dataset
    # Open http://localhost:8899
"""

import argparse
import json
import random
import re
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pyarrow.parquet as pq

# Global state
DATASET = []
REVIEWS = {}  # id → timestamp
COMMENTS = {}  # id → [{"text": ..., "timestamp": ...}]
DATA_DIR = None


def load_dataset(data_dir: Path):
    """Load metadata index only — texts loaded on demand."""
    global DATASET
    DATASET = []
    meta_cols = [c for c in ["id", "title", "author", "genre", "year_written",
                              "year_published", "text_length", "word_count",
                              "is_translation", "translator", "source",
                              "author_birth_year", "author_death_year",
                              "license", "date_text"] if True]
    for f in sorted(data_dir.glob("works-*.parquet")):
        t = pq.read_table(f, columns=[c for c in meta_cols if c in pq.read_schema(str(f)).names])
        for i in range(t.num_rows):
            rec = {"_file": str(f), "_row": i}
            for col in t.column_names:
                rec[col] = t.column(col)[i].as_py()
            DATASET.append(rec)
    print(f"Loaded {len(DATASET)} records (metadata only)")

    # Load saved reviews
    global REVIEWS, COMMENTS
    reviews_path = data_dir / "reviews.json"
    if reviews_path.exists():
        data = json.loads(reviews_path.read_text())
        REVIEWS = data.get("reviews", {})
        COMMENTS = data.get("comments", {})
        print(f"Loaded {len(REVIEWS)} reviews, {sum(len(v) for v in COMMENTS.values())} comments")


def save_reviews():
    path = DATA_DIR / "reviews.json"
    path.write_text(json.dumps({"reviews": REVIEWS, "comments": COMMENTS}, ensure_ascii=False, indent=2))


def get_random_record():
    rec = random.choice(DATASET)
    result = dict(rec)
    # Load text on demand from parquet
    f = Path(rec["_file"])
    row = rec["_row"]
    t = pq.read_table(f, columns=["text", "text_tagged"])
    result["text"] = t.column("text")[row].as_py() or ""
    result["text_tagged"] = t.column("text_tagged")[row].as_py() if "text_tagged" in t.column_names else ""
    del result["_file"]
    del result["_row"]
    result["_reviewed"] = REVIEWS.get(rec["id"], "")
    result["_comments"] = COMMENTS.get(rec["id"], [])
    return result


HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Корпус — Просмотрщик</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Georgia,serif;background:#f5f5f0;color:#333}
header{background:#8b0000;color:#fff;padding:12px 20px;display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;z-index:100}
header h1{font-size:1.1em}
.ctrls{display:flex;gap:8px;align-items:center}
.ctrls button{padding:7px 14px;border:none;border-radius:4px;cursor:pointer;font-size:0.85em;font-family:inherit}
.br{background:#fff;color:#8b0000;font-weight:bold}
.ba{background:#4caf50;color:#fff}
.bc{background:#ff9800;color:#fff}
.bt{background:#2196f3;color:#fff}
.bt.off{background:#90a4ae}
.bx{background:#eee;color:#333;font-size:0.8em}
.main{max-width:1000px;margin:15px auto;padding:0 15px}
.meta{background:#fff;border-radius:8px;padding:12px 18px;margin-bottom:12px;border:1px solid #ddd}
.mt{font-size:1.3em;color:#8b0000;font-weight:bold}
.ma{color:#555;font-style:italic;margin:3px 0}
.mi{color:#888;font-size:0.83em;display:flex;flex-wrap:wrap;gap:12px;margin-top:6px}
.badge{padding:2px 7px;border-radius:10px;font-size:0.78em;font-weight:bold}
.bp{background:#e8f5e9;color:#2e7d32}.bpo{background:#e3f2fd;color:#1565c0}.bd{background:#fce4ec;color:#c62828}
.bo{background:#f5f5f5;color:#616161}.bf{background:#fff3e0;color:#e65100}.bv{background:#c8e6c9;color:#1b5e20}
.tc{background:#fff;border-radius:8px;padding:15px;border:1px solid #ddd;margin-bottom:12px}
.tx{white-space:pre-wrap;line-height:1.7;font-size:0.93em;padding:8px}
.ts{background:#bbdefb;border-bottom:2px solid #1565c0;padding:1px 3px;border-radius:3px}
.tr{background:#f8bbd0;border-bottom:2px solid #c62828;padding:1px 3px;border-radius:3px;font-style:italic}
.tsc{background:#fff9c4;border-bottom:2px solid #f9a825;padding:2px 5px;border-radius:3px;font-weight:bold;display:inline-block;margin:5px 0}
.tsn{background:#1565c0;color:#fff;padding:1px 6px;border-radius:3px;font-weight:bold;font-size:0.88em}
.tsd{background:#7986cb;color:#fff;padding:1px 4px;border-radius:3px;font-size:0.83em}
.tl{font-size:0.6em;vertical-align:super;color:#666;margin-left:2px}
.cp{background:#fff;border-radius:8px;padding:12px 18px;border:1px solid #ff9800;margin-bottom:12px;display:none}
.cp.vis{display:block}
.cp textarea{width:100%;height:70px;padding:8px;font-family:inherit;font-size:0.88em;border:1px solid #ddd;border-radius:4px;resize:vertical}
.ca{margin-top:8px;display:flex;gap:8px}
.cs{background:#ff9800;color:#fff;padding:5px 12px;border:none;border-radius:4px;cursor:pointer}
.cc{background:#eee;color:#333;padding:5px 12px;border:none;border-radius:4px;cursor:pointer}
.st{background:#fff;border-radius:8px;padding:8px 18px;border:1px solid #ddd;margin-bottom:12px;font-size:0.83em;color:#666}
.ci{padding:3px 5px;margin:2px 0;background:#fff8e1;border-radius:3px;font-size:0.83em;border-bottom:1px solid #fff3e0}
.lg{display:flex;gap:12px;margin-bottom:10px;flex-wrap:wrap;font-size:0.83em}
.li{display:flex;align-items:center;gap:4px}
.ls{width:18px;height:12px;border-radius:3px;display:inline-block}
#cnt{font-weight:bold;margin-left:8px;color:#fff;font-size:0.85em}
</style>
</head>
<body>
<header>
<h1>📚 Корпус русской литературы</h1>
<div class="ctrls">
<button class="br" onclick="rand()">⟳ Случайная</button>
<button class="bt" id="tbtn" onclick="ttags()">Разметка: ВКЛ</button>
<button class="ba" onclick="approve()">✓ Проверено</button>
<button class="bc" onclick="tcomm()">✎ Коммент</button>
<button class="bx" onclick="xport()">💾 Экспорт</button>
<span id="cnt"></span>
</div>
</header>
<div class="main">
<div class="lg" id="lg" style="display:none">
<div class="li"><span class="ls" style="background:#1565c0"></span>Говорящий</div>
<div class="li"><span class="ls" style="background:#bbdefb;border-bottom:2px solid #1565c0"></span>Реплика</div>
<div class="li"><span class="ls" style="background:#f8bbd0;border-bottom:2px solid #c62828"></span>Ремарка</div>
<div class="li"><span class="ls" style="background:#fff9c4;border-bottom:2px solid #f9a825"></span>Сцена</div>
</div>
<div class="meta" id="meta" style="display:none">
<div class="mt" id="tit"></div>
<div class="ma" id="aut"></div>
<div class="mi" id="inf"></div>
</div>
<div class="cp" id="cpan">
<strong>Комментарий:</strong>
<textarea id="ctxt" placeholder="Опишите проблему..."></textarea>
<div class="ca"><button class="cs" onclick="scomm()">Сохранить</button><button class="cc" onclick="tcomm()">Отмена</button></div>
</div>
<div class="tc" id="tc" style="display:none"><div class="tx" id="txt"></div></div>
<div class="st" id="sp"><div id="slog"></div></div>
</div>
<script>
let cur=null,tags=true;
async function rand(){
  const r=await fetch('/api/random');
  cur=await r.json();
  render();
}
function render(){
  if(!cur)return;
  document.getElementById('meta').style.display='block';
  document.getElementById('tc').style.display='block';
  document.getElementById('lg').style.display=(tags&&cur.text_tagged)?'flex':'none';
  document.getElementById('tit').textContent=cur.title||cur.id;
  document.getElementById('aut').textContent=cur.author||'';
  let g=cur.genre||'other',gc={'prose':'bp','poetry':'bpo','drama':'bd','other':'bo','fable':'bf'}[g]||'bo';
  let inf='<span class="badge '+gc+'">'+g+'</span>';
  inf+='<span>'+(cur.text_length||0).toLocaleString()+' симв.</span>';
  if(cur.year_written)inf+='<span>написано: '+cur.year_written+'</span>';
  if(cur.year_published)inf+='<span>опубл.: '+cur.year_published+'</span>';
  if(cur.is_translation)inf+='<span>[перевод]</span>';
  if(cur.translator)inf+='<span>пер.: '+cur.translator+'</span>';
  if(cur.source)inf+='<span>источник: '+esc(cur.source).slice(0,60)+'</span>';
  if(cur.author_birth_year||cur.author_death_year)inf+='<span>автор: '+(cur.author_birth_year||'?')+'–'+(cur.author_death_year||'?')+'</span>';
  if(cur._reviewed)inf+=' <span class="badge bv">✓ '+cur._reviewed+'</span>';
  document.getElementById('inf').innerHTML=inf;
  const el=document.getElementById('txt');
  if(tags&&cur.text_tagged){el.innerHTML=rtags(cur.text_tagged)}
  else{el.textContent=cur.text||''}
  // Show comments
  let cl='';
  if(cur._comments&&cur._comments.length){
    for(const c of cur._comments)cl+='<div class="ci">✎ '+c.timestamp+': '+c.text+'</div>';
  }
  document.getElementById('slog').innerHTML=cl;
}
function rtags(t){
  let h=esc(t);
  h=h.replace(/&lt;scene&gt;(.*?)&lt;\/scene&gt;/g,'<span class="tsc">$1<span class="tl">сцена</span></span>');
  h=h.replace(/&lt;speaker name=&quot;([^&]+)&quot; dir=&quot;([^&]+)&quot;\/&gt;/g,'<span class="tsn">$1</span> <span class="tsd">($2)</span> ');
  h=h.replace(/&lt;speaker name=&quot;([^&]+)&quot;\/&gt;/g,'<span class="tsn">$1</span> ');
  h=h.replace(/&lt;stage&gt;(.*?)&lt;\/stage&gt;/g,'<span class="tr">$1</span>');
  return h;
}
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function ttags(){tags=!tags;document.getElementById('tbtn').textContent=tags?'Разметка: ВКЛ':'Разметка: ВЫКЛ';document.getElementById('tbtn').classList.toggle('off',!tags);render()}
async function approve(){if(!cur)return;const r=await fetch('/api/approve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:cur.id})});const d=await r.json();cur._reviewed=d.timestamp;render();updcnt()}
function tcomm(){document.getElementById('cpan').classList.toggle('vis')}
async function scomm(){if(!cur)return;const t=document.getElementById('ctxt').value.trim();if(!t)return;const r=await fetch('/api/comment',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:cur.id,text:t})});const d=await r.json();cur._comments=d.comments;document.getElementById('ctxt').value='';tcomm();render();updcnt()}
async function xport(){window.open('/api/export')}
async function updcnt(){const r=await fetch('/api/stats');const d=await r.json();document.getElementById('cnt').textContent=d.reviewed+'✓ '+d.comments+'✎'}
rand();updcnt();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode("utf-8"))
        elif path == "/api/random":
            rec = get_random_record()
            self.json_response(rec)
        elif path == "/api/stats":
            self.json_response({
                "reviewed": len(REVIEWS),
                "comments": sum(len(v) for v in COMMENTS.values()),
                "total": len(DATASET),
            })
        elif path == "/api/export":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Disposition", f'attachment; filename="corpus_review_{datetime.now().strftime("%Y-%m-%d")}.json"')
            self.end_headers()
            self.wfile.write(json.dumps({"reviews": REVIEWS, "comments": COMMENTS}, ensure_ascii=False, indent=2).encode("utf-8"))
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length)) if content_length else {}

        if path == "/api/approve":
            work_id = body.get("id", "")
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            REVIEWS[work_id] = ts
            save_reviews()
            self.json_response({"ok": True, "timestamp": ts})
        elif path == "/api/comment":
            work_id = body.get("id", "")
            text = body.get("text", "")
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            if work_id not in COMMENTS:
                COMMENTS[work_id] = []
            COMMENTS[work_id].append({"text": text, "timestamp": ts})
            save_reviews()
            self.json_response({"ok": True, "comments": COMMENTS[work_id]})
        else:
            self.send_error(404)

    def json_response(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, default=str).encode("utf-8"))

    def log_message(self, format, *args):
        pass  # Suppress request logging


def main():
    global DATA_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args()

    DATA_DIR = args.output_dir
    load_dataset(DATA_DIR)

    server = HTTPServer(("localhost", args.port), Handler)
    print(f"Viewer: http://localhost:{args.port}")
    print("Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()


if __name__ == "__main__":
    main()
