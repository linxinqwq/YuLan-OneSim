from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Tuple


def _tokenize(text: str) -> List[str]:
    s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text.lower(), flags=re.UNICODE)
    return [x for x in s.split() if len(x) > 1]


def _tf(text: str) -> Dict[str, float]:
    toks = _tokenize(text)
    if not toks:
        return {}
    n = len(toks)
    freq: Dict[str, int] = {}
    for t in toks:
        freq[t] = freq.get(t, 0) + 1
    return {k: v / n for k, v in freq.items()}


def cosine_tf_idf_sim(query: str, doc: str, corpus_docs: List[str]) -> float:
    q_tf = _tf(query)
    if not q_tf:
        return 0.0
    df: Dict[str, int] = {}
    doc_tfs = [_tf(d) for d in corpus_docs]
    vocab = set(q_tf.keys())
    for dt in doc_tfs:
        vocab |= set(dt.keys())
        for term in dt:
            df[term] = df.get(term, 0) + 1
    N = max(len(corpus_docs), 1)
    idf = {t: math.log((N + 1) / (df.get(t, 0) + 1)) + 1.0 for t in vocab}

    def vec(tf: Dict[str, float]) -> Dict[str, float]:
        return {t: tf.get(t, 0.0) * idf.get(t, 0.0) for t in vocab}

    qv, dv = vec(q_tf), vec(_tf(doc))
    dot = sum(qv[t] * dv[t] for t in vocab)
    nq = math.sqrt(sum(v * v for v in qv.values()))
    nd = math.sqrt(sum(v * v for v in dv.values()))
    if nq < 1e-9 or nd < 1e-9:
        return 0.0
    return dot / (nq * nd)


def match_subtasks_to_simulators(
    subtasks: List[Dict[str, Any]],
    manifest: List[Dict[str, Any]],
    min_score: float = 0.01,
) -> List[Dict[str, Any]]:
    stub_kw = ("参数初始化", "参数设置", "统计输出", "格式化输出", "打印结果")
    rich = [f"{m.get('category','')} {m.get('name','')} {m.get('description','')}" for m in manifest]
    results: List[Dict[str, Any]] = []
    for st in subtasks:
        desc = f"{st.get('title','')} {st.get('description','')}"
        if any(k in (st.get("title", "") + st.get("description", "")) for k in stub_kw):
            results.append(
                {
                    "subtask": st,
                    "matched_simulator": None,
                    "match_score": 0.0,
                    "needs_generated_stub": True,
                }
            )
            continue
        scored = [(cosine_tf_idf_sim(desc, d, rich), m) for m, d in zip(manifest, rich)]
        scored.sort(key=lambda x: -x[0])
        best_score, best = scored[0]
        if best_score < min_score:
            results.append(
                {
                    "subtask": st,
                    "matched_simulator": None,
                    "match_score": best_score,
                    "needs_generated_stub": True,
                }
            )
        else:
            results.append(
                {
                    "subtask": st,
                    "matched_simulator": best,
                    "match_score": best_score,
                    "needs_generated_stub": False,
                }
            )
    return results
