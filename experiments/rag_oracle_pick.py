"""Author-labelled supporting documents for the 20-question RAG diagnostic.

Local: python experiments/rag_oracle_pick.py

Candidates are the union of raw-query and prefixed-query dense top-20 over the
existing corpus, shown in corpus-index order without rank, score, or source so
the label is about the text, not the retriever. The reference answer IS shown
(author's decision, 2026-10-11): labels may lean toward reference similarity.
Documents outside the candidate pool cannot be picked; "no evidence" means "none
in this pool", not "none in the corpus".
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from dprag.chatdoctor import load_corpus
from dprag.config import ExperimentConfig
from rag_diagnose import DEFAULT_DIR, PREFIX, digest, documents, pin, read
from retrieval_probe import build_store

POOL = 20


def candidates(out: Path) -> dict:
    import torch

    retrieval = read(out / "retrieval.json")
    manifest = read(out / "dev_manifest.json")
    if retrieval["manifest_sha256"] != digest(manifest):
        raise SystemExit("retrieval.json does not match dev_manifest.json")
    cfg = ExperimentConfig(n_docs=manifest["n_docs"], corpus_seed=manifest["corpus_seed"],
                           embed_model=manifest["embed_model"])
    corpus = load_corpus(cfg.n_docs, cfg.corpus_seed)
    assert digest(corpus) == manifest["corpus_sha256"], "corpus content changed"
    store = build_store(cfg)
    embeddings = store.embeddings()
    rows = []
    for item in retrieval["rows"]:
        ranked = {}
        for label, prefix in (("raw", ""), ("prefix", PREFIX)):
            scores = (store.encode(prefix + item["query"]) @ embeddings.T)[0]
            values, indices = torch.topk(scores, POOL)
            ranked[label] = documents(corpus, zip(indices.tolist(), values.tolist()))
            # The frozen top-10 must be the head of this top-20, or the pool is not
            # the retrieval the generations were run on.
            frozen = [d["index"] for d in item["document_sets"][label + "_top10"]]
            if [d["index"] for d in ranked[label][:10]] != frozen:
                raise SystemExit(f"{item['id']} {label}: top-10 differs from retrieval.json")
        pool = sorted({d["index"] for docs in ranked.values() for d in docs})
        leaked = [i for i in pool if corpus[i].strip() == item["reference"].strip()]
        if leaked:
            raise SystemExit(f"{item['id']}: reference answer is itself in the pool {leaked}")
        rows.append({"id": item["id"], "query": item["query"], "reference": item["reference"],
                     "candidates": [{"index": i, "text": corpus[i]} for i in pool],
                     "rank": {k: [d["index"] for d in v] for k, v in ranked.items()}})
        print(f"{item['id']}: {len(pool)} candidates", flush=True)
    return {"schema": 1, "retrieval_sha256": digest(retrieval), "pool": POOL,
            "query_prefix": PREFIX, "rows": rows}


PAGE = """<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Oracle 文件標註</title>
<style>
:root{--bg:#fafaf9;--fg:#1c1917;--mute:#78716c;--card:#fff;--line:#e7e5e4;--sup:#dcfce7;--part:#fef9c3;--irr:#f5f5f4}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--fg:#e7e5e4;--mute:#a8a29e;--card:#292524;--line:#44403c;--sup:#14532d;--part:#713f12;--irr:#292524}}
body{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif;margin:0;padding:0 16px 80px}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 0;display:flex;gap:10px;flex-wrap:wrap;align-items:center;z-index:1}
section{max-width:900px;margin:24px auto}.box{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:8px 0;white-space:pre-wrap}
.doc[data-l=support]{background:var(--sup)}.doc[data-l=partial]{background:var(--part)}.doc[data-l=irrelevant]{background:var(--irr);opacity:.7}
.mute{color:var(--mute);font-size:13px}label{margin-right:12px;white-space:nowrap}textarea{width:100%;box-sizing:border-box;background:var(--card);color:var(--fg);border:1px solid var(--line)}
nav a{margin-right:6px}button{padding:4px 10px}
</style></head><body>
<header><b>Oracle 文件標註</b><span id="prog"></span>
<button onclick="exportJSON()">匯出 oracle_picks.json</button>
<label>匯入 <input type="file" accept=".json" onchange="importJSON(this.files[0])"></label></header>
<p class="mute" style="max-width:900px;margin:12px auto">判斷標準：這篇文件的內容，能否支持一個適用於<b>這位病患</b>的回答。
能支持＝有可直接採用的關鍵資訊；部分＝相關但需轉用或只涵蓋一部分；無關＝不適用。
文件依語料編號排序，不顯示檢索排名或來源。整題都沒有能支持的文件 → 勾「此候選池無證據」。
標註自動存在這個瀏覽器；完成後務必按「匯出」。</p>
<nav style="max-width:900px;margin:auto" id="nav"></nav>
__BODY__
<script>
const DATA=__DATA__;const KEY='oracle_picks_'+DATA.retrieval_sha256;
let S={labels:{},noev:{},notes:{}};
try{S=Object.assign(S,JSON.parse(localStorage.getItem(KEY)||'{}'))}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){}prog()}
function done(r){return S.noev[r.id]||r.candidates.some(c=>S.labels[r.id+':'+c.index]==='support')}
function prog(){const n=DATA.rows.filter(done).length;document.getElementById('prog').textContent=`完成 ${n}/${DATA.rows.length} 題`;
document.getElementById('nav').innerHTML=DATA.rows.map(r=>`<a href="#${r.id}">${done(r)?'✓':'·'}${r.id}</a>`).join('')}
function setL(id,idx,v){S.labels[id+':'+idx]=v;document.getElementById(id+'_'+idx).dataset.l=v;save()}
function restore(){for(const r of DATA.rows){document.getElementById('ne_'+r.id).checked=!!S.noev[r.id];
document.getElementById('nt_'+r.id).value=S.notes[r.id]||'';
for(const c of r.candidates){const v=S.labels[r.id+':'+c.index];const el=document.getElementById(r.id+'_'+c.index);el.dataset.l=v||'';
el.querySelectorAll('input').forEach(i=>i.checked=i.value===v)}}prog()}
function exportJSON(){const rows=DATA.rows.map(r=>({id:r.id,no_evidence:!!S.noev[r.id],note:S.notes[r.id]||'',
labels:Object.fromEntries(r.candidates.map(c=>[c.index,S.labels[r.id+':'+c.index]||null]))}));
const out={schema:1,retrieval_sha256:DATA.retrieval_sha256,exported_at:new Date().toISOString(),
complete:DATA.rows.every(done),rows};const a=document.createElement('a');
a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{type:'application/json'}));a.download='oracle_picks.json';a.click()}
function importJSON(f){f.text().then(t=>{const o=JSON.parse(t);if(o.retrieval_sha256!==DATA.retrieval_sha256){alert('候選池版本不符');return}
S={labels:{},noev:{},notes:{}};for(const r of o.rows){S.noev[r.id]=r.no_evidence;S.notes[r.id]=r.note;
for(const[k,v]of Object.entries(r.labels))if(v)S.labels[r.id+':'+k]=v}save();restore()})}
restore();
</script></body></html>"""


def render(out: Path, data: dict) -> None:
    sections = []
    for r in data["rows"]:
        docs = "".join(
            f'<div class="box doc" id="{r["id"]}_{c["index"]}"><div class="mute">#{c["index"]}　'
            + "".join(f'<label><input type="radio" name="{r["id"]}_{c["index"]}" value="{v}" '
                      f'onchange="setL(\'{r["id"]}\',{c["index"]},this.value)">{t}</label>'
                      for v, t in (("support", "能支持"), ("partial", "部分"), ("irrelevant", "無關")))
            + f'</div>{html.escape(c["text"])}</div>' for c in r["candidates"])
        sections.append(
            f'<section id="{r["id"]}"><h2>{r["id"]}</h2><div class="mute">病患問題</div>'
            f'<div class="box">{html.escape(r["query"])}</div><div class="mute">參考答案</div>'
            f'<div class="box">{html.escape(r["reference"])}</div>'
            f'<label><input type="checkbox" id="ne_{r["id"]}" onchange="S.noev[\'{r["id"]}\']=this.checked;save()">'
            f'此候選池無證據</label><div class="mute">備註</div><textarea id="nt_{r["id"]}" rows="2" '
            f'oninput="S.notes[\'{r["id"]}\']=this.value;save()"></textarea>'
            f'<div class="mute">候選文件 {len(r["candidates"])} 篇</div>{docs}</section>')
    lean = {"retrieval_sha256": data["retrieval_sha256"],
            "rows": [{"id": r["id"], "candidates": [{"index": c["index"]} for c in r["candidates"]]}
                     for r in data["rows"]]}
    page = (PAGE.replace("__BODY__", "".join(sections))
            .replace("__DATA__", json.dumps(lean).replace("</", "<\\/")))
    (out / "pick.html").write_text(page, encoding="utf-8")
    print("Pick page:", out / "pick.html", flush=True)


def main() -> None:
    out = DEFAULT_DIR
    path = out / "oracle_candidates.json"
    data = read(path) if path.exists() else candidates(out)
    pin(path, data)
    render(out, data)


if __name__ == "__main__":
    main()
