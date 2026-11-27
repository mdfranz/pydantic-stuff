"""Decision-model benchmark v2 for Ollama /v1/systemone models (pydantic-ai + Logfire).

Suites
  base     JevBench public set (231 items: original 72 / easy 48 / hard 111, MIT) — accuracy, chance-corrected
           accuracy, Brier, top-label ECE (10 bins), ordinal MAE, pydantic-ai rubric read rule, paraphrase
           consistency, trap (surface-answer) rate, latency, coverage (context/limit errors kept separate).
  perturb  Built from the short original+easy items:
             order_reverse / order_rotate   choice options reordered            -> answer flip rate
             anon_labels                    option keys -> neutral letters       -> answer flip rate
             none_option                    gold removed, "none" option added    -> picks none?
             polarity_swap                  yes/no meanings swapped              -> consistent flip?
             bundle                         2 unrelated questions added          -> answer change, |dp|
             repeat                         identical rerun                      -> determinism
  card     Gold option planted among N options (N = 8, 32, 64, 128, 255)     -> accuracy vs N

Usage:  uv run python decision_bench.py [--models m1 m2] [--suites base perturb card] [--limit N]
"""

import argparse
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import logfire

import s1_client as s1

ROOT = Path("/Users/mfranz/tmp/ollama-dms")
DATA = ROOT / "benchmarks" / "data" / "jevbench"
OUT = ROOT / "benchmarks"
MODELS = [
    "laya:latest",
    "d1:2b-mxfp8",
    "strands-decider:2b-mxfp8",
    "d3:4b-mxfp8",
    "kev:9b-nvfp4",
    "d3:9b-nvfp4",
    "nimble-v3:9b-nvfp4",
]
CARD_SIZES = [8, 32, 64, 128, 255]
CARD_ITEMS = 12
SEED = 20261126

NONE_KEY = "none_of_these"
NONE_DESC = "None of the listed options fits this case."
BUNDLE_EXTRAS = {
    "x_tone": {"type": "noul", "instructions": "Is the writer's tone negative or frustrated?"},
    "x_length": {"type": "score", "instructions": "How long is the text?", "criteria": ["Short", "Medium", "Long"]},
}


# ---------------------------------------------------------------- data


def load_items() -> list[dict]:
    items = []
    for tier in ("original", "easy", "hard"):
        for line in (DATA / f"{tier}.jsonl").read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                item["tier"] = tier
                items.append(item)
    return items


def gold_of(item: dict) -> str:
    """Gold answer as a label key: choice key, 'yes'/'no', or score level as a string."""
    exp = item["expected"]
    if item["question"]["type"] == "noul":
        return "yes" if str(exp).lower() in ("yes", "true") else "no"
    return str(exp)


def labels_of(question: dict) -> list[str]:
    t = question["type"]
    if t == "noul":
        return ["no", "yes"]
    if t == "choice":
        return list(question["criteria"].keys())
    return [str(i) for i in range(len(question["criteria"]))]


# ---------------------------------------------------------------- answer reading


def read_answer(question: dict, ans: dict) -> dict:
    """Normalize one answer to {dist, pred, score?}. dist is over labels_of(question)."""
    t = question["type"]
    labels = labels_of(question)
    if t == "noul":
        p = float(ans["noul"])
        dist = {"no": 1.0 - p, "yes": p}
    else:
        raw = {str(k): float(v) for k, v in (ans.get("probabilities") or {}).items()}
        dist = {k: raw.get(k, 0.0) for k in labels}
    pred = max(labels, key=lambda k: dist[k])  # ties -> first in option order
    out: dict[str, Any] = {"dist": dist, "pred": pred}
    if t == "score":
        score = float(ans["score"])
        out["score"] = score
        out["pai_pred"] = str(min(int(score + 0.5), len(labels) - 1))  # pydantic-ai read rule
    return out


def run_one(model: str, state: Any, question: dict, extras: dict | None = None) -> dict:
    questions = {"q": question, **(extras or {})}
    resp, ms = s1.ask(model, state, questions)
    rec: dict[str, Any] = {"ms": round(ms, 1)}
    if "error" in resp:
        rec["error"] = str(resp["error"])[:300]
        return rec
    try:
        rec.update(read_answer(question, resp["answers"]["q"]))
        rec["input_tokens"] = (resp.get("usage") or {}).get("input_tokens")
    except (KeyError, TypeError, ValueError) as e:
        rec["error"] = f"malformed: {type(e).__name__}: {e}"
    return rec


# ---------------------------------------------------------------- perturbations


def perturb_cases(items: list[dict]) -> list[dict]:
    """Variants of short items. Each: {kind, base_id, state, question, gold, extras?}."""
    cases = []
    for it in items:
        if it["tier"] not in ("original", "easy"):
            continue
        q, gold = it["question"], gold_of(it)
        base = {"base_id": it["id"], "state": it["state"]}
        if q["type"] == "choice":
            crit = q["criteria"]
            keys = list(crit)
            rev = dict(reversed(list(crit.items())))
            rot = {k: crit[k] for k in keys[1:] + keys[:1]}
            cases.append({**base, "kind": "order_reverse", "question": {**q, "criteria": rev}, "gold": gold})
            cases.append({**base, "kind": "order_rotate", "question": {**q, "criteria": rot}, "gold": gold})
            letters = {k: f"option_{chr(65 + i)}" for i, k in enumerate(keys)}
            anon = {letters[k]: v for k, v in crit.items()}
            cases.append({**base, "kind": "anon_labels", "question": {**q, "criteria": anon}, "gold": letters[gold]})
            if len(keys) >= 3 and not any(w in k.lower() for k in keys for w in ("other", "none")):
                dropped = {k: v for k, v in crit.items() if k != gold}
                dropped[NONE_KEY] = NONE_DESC
                cases.append({**base, "kind": "none_option", "question": {**q, "criteria": dropped}, "gold": NONE_KEY})
        if q["type"] == "noul" and q.get("criteria"):
            c = q["criteria"]
            swapped = {**q, "criteria": {"true": c["false"], "false": c["true"]}}
            cases.append({**base, "kind": "polarity_swap", "question": swapped, "gold": "no" if gold == "yes" else "yes"})
        cases.append({**base, "kind": "bundle", "question": q, "gold": gold, "extras": BUNDLE_EXTRAS})
        if it["tier"] == "original":
            cases.append({**base, "kind": "repeat", "question": q, "gold": gold})
    return cases


def card_cases(items: list[dict]) -> list[dict]:
    """Gold planted among N options; distractors drawn from other items' option labels."""
    rng = random.Random(SEED)
    choice = [i for i in items if i["question"]["type"] == "choice"]
    pool: dict[str, str] = {}
    for it in choice:
        for k, v in it["question"]["criteria"].items():
            pool.setdefault(k, v)
    picks = [i for i in choice if i["tier"] in ("original", "easy") and i.get("family") in ("intent", "routing", "tool_selection")]
    rng.shuffle(picks)
    cases = []
    for it in picks[:CARD_ITEMS]:
        own = it["question"]["criteria"]
        candidates = [k for k in pool if k not in own]
        for n in CARD_SIZES:
            extra = rng.sample(candidates, min(n - len(own), len(candidates)))
            opts = list(own.items()) + [(k, pool[k]) for k in extra]
            rng.shuffle(opts)
            q = {**it["question"], "criteria": dict(opts)}
            cases.append({"kind": f"card_{n}", "n": len(opts), "base_id": it["id"], "state": it["state"], "question": q, "gold": gold_of(it)})
    return cases


# ---------------------------------------------------------------- metrics


def brier(dist: dict[str, float], gold: str) -> float:
    return sum((p - (1.0 if k == gold else 0.0)) ** 2 for k, p in dist.items())


def ece(records: list[dict], bins: int = 10) -> float | None:
    pts = [(max(r["dist"].values()), r["pred"] == r["gold"]) for r in records if "dist" in r]
    if not pts:
        return None
    total, err = len(pts), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(c, ok) for c, ok in pts if (lo < c <= hi) or (b == 0 and c == 0)]
        if sel:
            err += len(sel) / total * abs(statistics.mean(ok for _, ok in sel) - statistics.mean(c for c, _ in sel))
    return err


def pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}"


def num(x: float | None, nd: int = 3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def summarize_base(recs: list[dict]) -> dict:
    n = len(recs)
    valid = [r for r in recs if "pred" in r]
    correct = [r for r in valid if r["pred"] == r["gold"]]
    chance = statistics.mean(1 / r["n_labels"] for r in recs) if recs else 0
    acc = len(correct) / n if n else None
    s: dict[str, Any] = {
        "n": n,
        "coverage": len(valid) / n if n else None,
        "acc": acc,
        "acc_valid": len(correct) / len(valid) if valid else None,
        "cc_acc": max(0.0, (acc - chance) / (1 - chance)) if acc is not None else None,
        "brier": statistics.mean(brier(r["dist"], r["gold"]) for r in valid) if valid else None,
        "ece": ece(valid),
        "p50_ms": statistics.median(r["ms"] for r in valid) if valid else None,
        "p95_ms": sorted(r["ms"] for r in valid)[int(0.95 * (len(valid) - 1))] if valid else None,
    }
    for tier in ("original", "easy", "hard"):
        t = [r for r in recs if r["tier"] == tier]
        s[f"acc_{tier}"] = sum(1 for r in t if r.get("pred") == r["gold"]) / len(t) if t else None
    scores = [r for r in valid if "score" in r]
    if scores:
        s["ord_mae"] = statistics.mean(abs(r["score"] - int(r["gold"])) for r in scores)
        s["score_acc_argmax"] = sum(r["pred"] == r["gold"] for r in scores) / len(scores)
        s["score_acc_pai"] = sum(r["pai_pred"] == r["gold"] for r in scores) / len(scores)
        s["n_score"] = len(scores)
    groups = defaultdict(list)
    for r in recs:
        if r["tier"] == "original":
            groups[r["group"]].append(r)
    pairs = [g for g in groups.values() if len(g) >= 2 and all("pred" in r for r in g)]
    if pairs:
        s["para_same"] = sum(len({r["pred"] for r in g}) == 1 for g in pairs) / len(pairs)
        s["para_both"] = sum(all(r["pred"] == r["gold"] for r in g) for g in pairs) / len(pairs)
    traps = [r for r in recs if r.get("surface") is not None]
    if traps:
        s["trap_rate"] = sum(r.get("pred") == r["surface"] for r in traps) / len(traps)
        s["n_trap"] = len(traps)
    fam = defaultdict(list)
    for r in recs:
        if r["tier"] == "hard":
            fam[r["family"]].append(r)
    s["hard_family_acc"] = {f: sum(r.get("pred") == r["gold"] for r in rs) / len(rs) for f, rs in sorted(fam.items())}
    s["errors"] = Counter(r["error"].split(":")[0][:80] for r in recs if "error" in r)
    return s


def summarize_perturb(recs: list[dict], base_by_id: dict[str, dict]) -> dict:
    s: dict[str, Any] = {}
    by_kind = defaultdict(list)
    for r in recs:
        by_kind[r["kind"]].append(r)
    for kind, rs in by_kind.items():
        ok = [r for r in rs if "pred" in r and "pred" in base_by_id.get(r["base_id"], {})]
        if not ok:
            s[kind] = {"n": len(rs), "valid": 0}
            continue
        entry: dict[str, Any] = {"n": len(rs), "valid": len(ok), "acc": sum(r["pred"] == r["gold"] for r in ok) / len(ok)}
        if kind in ("order_reverse", "order_rotate", "bundle", "repeat"):
            entry["flip"] = sum(r["pred"] != base_by_id[r["base_id"]]["pred"] for r in ok) / len(ok)
            entry["mean_abs_dp_gold"] = statistics.mean(
                abs(r["dist"].get(r["gold"], 0) - base_by_id[r["base_id"]]["dist"].get(r["gold"], 0)) for r in ok
            )
        if kind == "anon_labels":
            # Same option chosen as in the base run, mapped through the letter renaming.
            def same(r: dict) -> bool:
                base_q = next(i for i in ITEMS if i["id"] == r["base_id"])["question"]["criteria"]
                letters = {k: f"option_{chr(65 + i)}" for i, k in enumerate(base_q)}
                return letters[base_by_id[r["base_id"]]["pred"]] == r["pred"]

            entry["flip"] = sum(not same(r) for r in ok) / len(ok)
        if kind == "polarity_swap":
            entry["consistent"] = sum(r["pred"] != base_by_id[r["base_id"]]["pred"] for r in ok) / len(ok)
        s[kind] = entry
    return s


def summarize_card(recs: list[dict]) -> dict:
    out = {}
    for n in CARD_SIZES:
        rs = [r for r in recs if r["kind"] == f"card_{n}"]
        valid = [r for r in rs if "pred" in r]
        out[n] = {
            "n": len(rs),
            "valid": len(valid),
            "acc": sum(r.get("pred") == r["gold"] for r in rs) / len(rs) if rs else None,
            "p50_ms": statistics.median(r["ms"] for r in valid) if valid else None,
        }
    return out


# ---------------------------------------------------------------- run


ITEMS: list[dict] = []


def run_model(model: str, suites: list[str], limit: int | None) -> dict:
    s1.unload_all()
    warm_ms = s1.warmup(model)
    res = s1.resident_info(model)
    print(f"\n=== {model} | warmup {warm_ms:.0f} ms | resident {res}", flush=True)
    out: dict[str, Any] = {"model": model, "warmup_ms": round(warm_ms, 1), "resident": res}
    base_by_id: dict[str, dict] = {}

    with logfire.span("decision_bench {model}", model=model):
        if "base" in suites or "perturb" in suites:
            items = ITEMS if limit is None else ITEMS[:limit]
            recs = []
            with logfire.span("suite base", model=model):
                for i, it in enumerate(items):
                    r = run_one(model, it["state"], it["question"])
                    r.update(
                        id=it["id"], tier=it["tier"], family=it.get("family"), group=it.get("group"),
                        type=it["question"]["type"], n_labels=len(labels_of(it["question"])), gold=gold_of(it),
                        surface=(None if it["provenance"].get("surface_answer") is None else str(it["provenance"]["surface_answer"])),
                    )
                    recs.append(r)
                    base_by_id[it["id"]] = r
                    if (i + 1) % 25 == 0:
                        print(f"  base {i + 1}/{len(items)}", flush=True)
            out["base_records"] = recs
            out["base"] = summarize_base(recs)
            b = out["base"]
            print(f"  base: acc {pct(b['acc'])} | cc {pct(b['cc_acc'])} | cov {pct(b['coverage'])} | "
                  f"brier {num(b['brier'])} | ece {num(b['ece'])} | p50 {num(b['p50_ms'], 0)} ms | errors {dict(b['errors'])}", flush=True)

        if "perturb" in suites:
            cases = perturb_cases(ITEMS if limit is None else ITEMS[:limit])
            recs = []
            with logfire.span("suite perturb", model=model):
                for c in cases:
                    r = run_one(model, c["state"], c["question"], c.get("extras"))
                    r.update(kind=c["kind"], base_id=c["base_id"], gold=c["gold"])
                    recs.append(r)
            out["perturb_records"] = recs
            out["perturb"] = summarize_perturb(recs, base_by_id)
            print("  perturb: " + " | ".join(f"{k} {json.dumps({kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()})}" for k, v in out["perturb"].items()), flush=True)

        if "card" in suites:
            cases = card_cases(ITEMS)
            recs = []
            with logfire.span("suite card", model=model):
                for c in cases:
                    r = run_one(model, c["state"], c["question"])
                    r.update(kind=c["kind"], n=c["n"], base_id=c["base_id"], gold=c["gold"])
                    recs.append(r)
            out["card_records"] = recs
            out["card"] = summarize_card(recs)
            print("  card: " + " | ".join(f"N={n}: {pct(v['acc'])}% ({v['valid']}/{v['n']} valid)" for n, v in out["card"].items()), flush=True)

    s1.unload_all()
    return out


def write_report(data: dict, md_path: Path) -> None:
    ms = data["models"]
    L = [
        f"# Decision-Model Benchmark v2 — Mac Mini M4 24GB (`{data['timestamp']}`)",
        "",
        f"Harness: `pai-poc/decision_bench.py` (pydantic-ai exact-payload `/v1/systemone` model + Logfire). Ollama `{data.get('ollama_version')}`. "
        "Data: JevBench public set (MIT, `benchmarks/data/jevbench`). Errors (e.g. context overflow) count as wrong in `Acc`; "
        "`Acc (valid)` excludes them. Single run, temperature not settable on `/v1/systemone`. `[Measured]`",
        "",
    ]
    if any("base" in m for m in ms):
        L += ["## Accuracy & Calibration (JevBench public, 231 items)", "",
              "| Model | Resident GB | Ctx | Coverage | Acc | Acc (valid) | Chance-corr. | Original | Easy | Hard | Brier ↓ | ECE ↓ | Ord. MAE ↓ | Score acc argmax / pai-round | Paraphrase same / both ✓ | Trap rate ↓ | p50 / p95 ms |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for m in ms:
            b = m.get("base")
            if not b:
                continue
            L.append(
                f"| `{m['model']}` | {m['resident']['size_gb']} | {m['resident']['context_length']} | {pct(b['coverage'])} | {pct(b['acc'])} | {pct(b['acc_valid'])} | "
                f"{pct(b['cc_acc'])} | {pct(b['acc_original'])} | {pct(b['acc_easy'])} | {pct(b['acc_hard'])} | {num(b['brier'])} | {num(b['ece'])} | "
                f"{num(b.get('ord_mae'), 2)} | {pct(b.get('score_acc_argmax'))} / {pct(b.get('score_acc_pai'))} | {pct(b.get('para_same'))} / {pct(b.get('para_both'))} | "
                f"{pct(b.get('trap_rate'))} | {num(b['p50_ms'], 0)} / {num(b['p95_ms'], 0)} |")
        fams = sorted({f for m in ms if m.get("base") for f in m["base"]["hard_family_acc"]})
        if fams:
            L += ["", "### Hard tier by family (accuracy %)", "", "| Model | " + " | ".join(fams) + " |", "| --- |" + " ---: |" * len(fams)]
            for m in ms:
                if m.get("base"):
                    fa = m["base"]["hard_family_acc"]
                    L.append(f"| `{m['model']}` | " + " | ".join(pct(fa.get(f)) for f in fams) + " |")
        errs = [(m["model"], m["base"]["errors"]) for m in ms if m.get("base") and m["base"]["errors"]]
        if errs:
            L += ["", "### Errors (base suite)", ""] + [f"- `{mod}`: " + "; ".join(f"{k} ×{v}" for k, v in e.items()) for mod, e in errs]
    if any("perturb" in m for m in ms):
        L += ["", "## Robustness (perturbations of original + easy items)", "",
              "| Model | Order flip (rev / rot) ↓ | Anon-label flip ↓ | None-option acc ↑ | Yes/no meaning swap followed ↑ | Bundle flip ↓ / mean abs Δp | Repeat flip ↓ / mean abs Δp |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for m in ms:
            p = m.get("perturb")
            if not p:
                continue
            g = lambda k, f: p.get(k, {}).get(f)  # noqa: E731
            L.append(
                f"| `{m['model']}` | {pct(g('order_reverse', 'flip'))} / {pct(g('order_rotate', 'flip'))} | {pct(g('anon_labels', 'flip'))} | "
                f"{pct(g('none_option', 'acc'))} | {pct(g('polarity_swap', 'consistent'))} | {pct(g('bundle', 'flip'))} / {num(g('bundle', 'mean_abs_dp_gold'))} | "
                f"{pct(g('repeat', 'flip'))} / {num(g('repeat', 'mean_abs_dp_gold'))} |")
    if any("card" in m for m in ms):
        L += ["", f"## Option-count scaling ({CARD_ITEMS} intent/routing items; accuracy %, errors count wrong)", "",
              "| Model | " + " | ".join(f"N={n}" for n in CARD_SIZES) + " | p50 ms @ N=255 |", "| --- |" + " ---: |" * (len(CARD_SIZES) + 1)]
        for m in ms:
            c = m.get("card")
            if c:
                L.append(f"| `{m['model']}` | " + " | ".join(f"{pct(c[n]['acc'])} ({c[n]['valid']}/{c[n]['n']})" for n in CARD_SIZES) + f" | {num(c[255]['p50_ms'], 0)} |")
    md_path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--suites", nargs="*", default=["base", "perturb", "card"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N items (smoke test)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    s1.configure_logfire("decision-bench-v2")
    ITEMS.extend(load_items())
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    stem = f"decision_bench_v2_{stamp}{('_' + args.tag) if args.tag else ''}"
    data: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "ollama_version": s1.httpx.get(f"{s1.OLLAMA}/api/version", timeout=10).json().get("version"),
        "suites": args.suites,
        "limit": args.limit,
        "models": [],
    }
    for model in args.models:
        data["models"].append(run_model(model, args.suites, args.limit))
        (OUT / f"{stem}.json").write_text(json.dumps(data, indent=1, default=lambda o: dict(o) if isinstance(o, Counter) else str(o)))
        write_report(data, OUT / f"{stem}.md")
    print(f"\nWrote {OUT / stem}.json and .md")


if __name__ == "__main__":
    main()
