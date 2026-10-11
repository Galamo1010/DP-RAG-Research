"""Score the frozen ordinary-RAG diagnostic locally, after generation finishes.

python experiments/rag_diagnose_score.py --bertscore

Metrics are diagnostic: overlap is not clinical correctness, exact copying is
not a privacy test, and changing the ranked documents is not proven relevance.
Paired intervals use queries as units; n=20 is exploratory, not confirmatory.
"""

from __future__ import annotations

import argparse
import difflib
import html
import json
import re
import statistics
from pathlib import Path

from dprag.chatdoctor import reference_damage
from dprag.quality import bertscore_f1, rouge_l_f1
from rag_diagnose import ARMS, DEFAULT_DIR, digest, read, write

WORDS = re.compile(r"[A-Za-z0-9]+")
COMPARISONS = [
    ("rag_old_dp128", "norag_old128", "舊 prompt：加入原檢索文件"),
    ("norag_task128", "norag_old128", "NoRAG：只改任務指令"),
    ("rag_task_dp128", "rag_old_dp128", "固定原檢索文件：只改任務指令"),
    ("rag_task_prefix5_128", "rag_task_raw5_128", "固定前五篇規則：只改 query 前綴"),
    ("rag_task_prefix10_128", "rag_task_prefix5_128", "同一排序：5 篇改 10 篇"),
    ("rag_task_prefix5_256", "rag_task_prefix5_128", "同一輸入：128 上限改 256"),
    ("rag_task_prefix5_128", "norag_task128", "新版 RAG 與新版 NoRAG"),
]


def interval(values: list[float]) -> dict:
    mean = statistics.mean(values)
    half = 1.96 * statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else None
    return {"mean": mean, "low": mean - half if half is not None else None,
            "high": mean + half if half is not None else None, "n": len(values)}


def longest_copy(text: str, documents: list[dict]) -> int:
    words = WORDS.findall(text.lower())
    return max((difflib.SequenceMatcher(None, words, WORDS.findall(d['text'].lower()),
                                       autojunk=False).find_longest_match().size
                for d in documents), default=0)


def score(out: Path, use_bert: bool) -> None:
    manifest = read(out / "retrieval.json")
    generated = read(out / "generation.json")
    if generated["params"]["manifest_sha256"] != digest(manifest):
        raise SystemExit("Evidence manifest does not match the generated answers")
    cases = {item["id"]: item for item in manifest["rows"]}
    keyed = {(r["id"], r["arm"]): r for r in generated["rows"]}
    expected = {(i, a) for i in cases for a in ARMS}
    if set(keyed) != expected or len(keyed) != len(generated["rows"]):
        raise SystemExit(f"Need one result for every case/arm; missing={len(expected - set(keyed))}, "
                         f"unexpected={len(set(keyed) - expected)}, duplicate rows="
                         f"{len(generated['rows']) - len(keyed)}")
    keys = sorted(keyed)
    fingerprint = digest(generated)
    model = "microsoft/deberta-xlarge-mnli"
    cache_path = out / "bertscore.json"
    cache = read(cache_path) if cache_path.exists() else {}
    cache_matches = (cache.get("generation_sha256") == fingerprint
                     and cache.get("model") == model and cache.get("rescale") is True
                     and cache.get("keys") == [list(k) for k in keys]
                     and len(cache.get("scores", [])) == len(keys))
    if use_bert and not cache_matches:
        print(f"Scoring {len(keys)} pairs locally with BERTScore (batch size 4)", flush=True)
        values = bertscore_f1([keyed[k]["text"] for k in keys],
                             [cases[k[0]]["reference"] for k in keys],
                             model=model, rescale=True, batch_size=4)
        cache = {"generation_sha256": fingerprint, "model": model,
                 "rescale": True, "keys": keys, "scores": values}
        write(cache_path, cache)
        cache_matches = True
    bert = dict(zip(keys, cache["scores"])) if cache_matches else {}
    measured = {}
    for key in keys:
        row, case = keyed[key], cases[key[0]]
        doc_set, _, _ = ARMS[key[1]]
        docs = [] if doc_set == 'none' else case['document_sets'][doc_set]
        lower = row['text'].lower()
        measured[key] = {
            "id": key[0], "arm": key[1],
            "rouge_l": rouge_l_f1(row['text'], case['reference']),
            "bertscore": bert.get(key), "hit_limit": row['hit_limit'],
            "words": len(WORDS.findall(row['text'])), "tokens": len(row['emitted']),
            "platform_or_link": bool(re.search(r'chat\s*doctor|https?://|bit\.ly', lower)),
            "longest_exact_copy_words": longest_copy(row['text'], docs),
            "reference_damage": sorted(reference_damage(case['reference'])),
            "seconds": row['seconds'],
        }
    arms = {}
    for arm in ARMS:
        rows = [measured[(i, arm)] for i in cases]
        arms[arm] = {
            "n": len(rows), "rouge_l": interval([r['rouge_l'] for r in rows]),
            "bertscore": interval([r['bertscore'] for r in rows]) if bert else None,
            "hit_limit_count": sum(r['hit_limit'] for r in rows),
            "platform_or_link_count": sum(r['platform_or_link'] for r in rows),
            "mean_words": statistics.mean(r['words'] for r in rows),
            "mean_seconds": statistics.mean(r['seconds'] for r in rows),
            "mean_longest_copy": statistics.mean(r['longest_exact_copy_words'] for r in rows),
        }
    pairs = []
    for treatment, control, reason in COMPARISONS:
        result = {"treatment": treatment, "control": control, "question": reason}
        for metric in ['rouge_l', 'bertscore'] if bert else ['rouge_l']:
            result[metric] = interval([measured[(i, treatment)][metric] - measured[(i, control)][metric]
                                       for i in cases])
        result['byte_identical_count'] = sum(keyed[(i, treatment)]['text'] == keyed[(i, control)]['text']
                                              for i in cases)
        pairs.append(result)
    result = {"generation_sha256": fingerprint,
              "note": "Exploratory n=20; query-paired normal 95% CIs, no multiplicity correction. "
                      "Scores measure reference similarity, not medical correctness or evidence support.",
              "damaged_reference_queries": sum(bool(reference_damage(c['reference'])) for c in cases.values()),
              "arms": arms, "paired": pairs, "per_item": list(measured.values())}
    write(out / 'analysis.json', result)
    rows_html = []
    print('arm | n | ROUGE-L | BERTScore | cap | platform/link | words | sec', flush=True)
    for arm, values in arms.items():
        b = f"{values['bertscore']['mean']:.4f}" if values['bertscore'] else 'not scored'
        cells = [arm, str(values['n']), f"{values['rouge_l']['mean']:.4f}", b,
                 str(values['hit_limit_count']), str(values['platform_or_link_count']),
                 f"{values['mean_words']:.1f}", f"{values['mean_seconds']:.1f}"]
        print(' | '.join(cells))
        rows_html.append('<tr>' + ''.join('<td>' + html.escape(c) + '</td>' for c in cells) + '</tr>')
    pairs_html = []
    for pair in pairs:
        cells = [pair['question']]
        for metric in ['rouge_l', 'bertscore']:
            if metric in pair:
                value = pair[metric]
                cells.append(f"{value['mean']:+.4f} [{value['low']:+.4f}, {value['high']:+.4f}]")
            else:
                cells.append('not scored')
        print(pair['treatment'], '-', pair['control'], ' | '.join(cells[1:]))
        pairs_html.append('<tr>' + ''.join('<td>' + html.escape(c) + '</td>' for c in cells) + '</tr>')
    page = '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>RAG 診斷結果</title>
<style>body{font:16px/1.6 system-ui;max-width:1200px;margin:30px auto;padding:15px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ddd;text-align:left}th{background:#eee}p{max-width:1000px}</style>
<h1>20 題 RAG 診斷結果</h1><p>每組使用相同 20 題。cap 表示到達 token 上限且未輸出終止 token。
平台／連結只計字串命中，不是醫療錯誤率。ROUGE-L／BERTScore 衡量參考答案相似度，不代表回答正確或有依據。</p>
<p><a href="review.html">開啟逐題回答與原始文件核對</a></p>
<table><tr><th>arm</th><th>n</th><th>ROUGE-L</th><th>BERTScore</th><th>cap</th><th>平台／連結</th><th>平均字數</th><th>秒／題</th></tr>
''' + ''.join(rows_html) + '''</table><h2>配對差（前者減後者）</h2>
<p>探索性 20 題、以問題為統計單位的常態近似 95% CI，未做多重比較校正；不作非劣性宣稱。</p>
<table><tr><th>比較目的</th><th>ROUGE-L 差 [95% CI]</th><th>BERTScore 差 [95% CI]</th></tr>
''' + ''.join(pairs_html) + '</table></html>'
    (out / 'analysis.html').write_text(page, encoding='utf-8')
    print('Analysis:', out / 'analysis.html', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_DIR)
    parser.add_argument('--bertscore', action='store_true')
    args = parser.parse_args()
    score(args.out, args.bertscore)
