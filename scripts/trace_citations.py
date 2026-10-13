#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
trace_citations.py -- citation trace (Semantic Scholar first, then OpenAlex).

MILESTONE version (v5, 2026-10-03)
----------------------------------
Changes over v4, made so that the final run is *verifiable*:

* COMPLETENESS TRACKING. v4 aborted a whole (seed, direction) on a persistent
  HTTP error but still wrote the partial rows -- a milestone could silently be
  missing half of a 2000-row seed. Now every (seed, direction) records
  {pages, items, aborted, reason}, and the run prints a COMPLETENESS table.
  A run is only a valid milestone if every line says `ok`.
* ERROR LOG NOT TRUNCATED. v4 kept only the first 20 errors; now up to 500.
* `--seeds S1,S3` : re-run only selected seeds (fix a single incomplete
  direction without redoing the whole trace).
* OpenAlex NOTE: as of 2026-10-03 OpenAlex's record for the primary seed
  (W2795843265) is poisoned -- its title/abstract belong to an unrelated
  "VOLE-Field" paper, its citation count (175) is 12x below Semantic Scholar
  (2143), and a manual check of one listed citing work found no reference to
  the seed. Recommendation: run with `dbs: [semantic_scholar]` in the config.

Auth (env only; never written to output)
  $env:S2_API_KEY       = "<semantic scholar key>"
  $env:OPENALEX_API_KEY = "<openalex key>"        # only if OpenAlex is enabled
  $env:OPENALEX_EMAIL   = "you@example.com"

Usage
  python scripts/trace_citations.py --config configs/seed_papers.yaml --selftest
  python scripts/trace_citations.py --config configs/seed_papers.yaml --resolve-only
  python scripts/trace_citations.py --config configs/seed_papers.yaml --mode snowball
  python scripts/trace_citations.py --config configs/seed_papers.yaml --seeds S1,S3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from datetime import date

import urllib.parse

import requests
import yaml

S2_BASE = "https://api.semanticscholar.org/graph/v1"
OA_BASE = "https://api.openalex.org"

S2_API_KEY = os.environ.get("S2_API_KEY") or ""
OA_API_KEY = os.environ.get("OPENALEX_API_KEY") or ""
OA_EMAIL = os.environ.get("OPENALEX_EMAIL") or ""
try:
    S2_SLEEP = float(os.environ.get("TRACE_S2_SLEEP", "3.0"))
except ValueError:
    S2_SLEEP = 3.0
OA_SLEEP = 0.15
RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 6

CSV_FIELDS = [
    "date", "db", "direction", "seed_id", "query", "title", "year", "venue",
    "authors", "arxiv_id", "doi", "s2_paper_id", "openalex_id",
    "matched_keywords", "keyword_group", "hits", "decision", "exclude_reason",
    "abstract_missing", "verify_venue",
]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
class Limiter:
    def __init__(self, min_interval):
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self):
        dt = time.time() - self._last
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last = time.time()


class Ctx:
    """Holds outputs so they can be persisted even if the run raises."""

    def __init__(self):
        self.rows = []
        self.status_counter = {}
        self.errors = []
        self.per_seed = {}
        self.completeness = {}     # seed_id -> {direction: {...}}
        self.unresolved = []
        self.resolved_by_title = []
        self.requests = 0

    def bump(self, db, status):
        self.status_counter[f"{db}:{status}"] = self.status_counter.get(f"{db}:{status}", 0) + 1

    def err(self, msg):
        if len(self.errors) < 500:
            self.errors.append(msg)
        print("  ! " + msg)

    def note_complete(self, seed_id, direction, pages, items, aborted, reason=""):
        self.completeness.setdefault(seed_id, {})[direction] = {
            "pages": pages, "items": items,
            "aborted": bool(aborted), "reason": reason,
        }


def strip_wild(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"[*?]+", " ", s).strip()


def norm_title(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def title_sim(a: str, b: str) -> float:
    ta, tb = set(norm_title(a).split()), set(norm_title(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / float(len(ta | tb))


def backoff(attempt):
    return min(30.0, (2 ** attempt) + random.uniform(0, 0.7))


def short(s, n=90):
    s = (s or "").replace("\n", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


# --------------------------------------------------------------------------- #
# HTTP wrappers with retry
# --------------------------------------------------------------------------- #
_s2_lim = Limiter(S2_SLEEP)
_oa_lim = Limiter(OA_SLEEP)


def s2_get(path, params):
    url = S2_BASE + path
    headers = {}
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY
    last = None
    for attempt in range(MAX_ATTEMPTS):
        _s2_lim.wait()
        try:
            r = requests.get(url, params=params, headers=headers, timeout=40)
        except requests.RequestException as e:
            last = ("ERR", str(e))
            time.sleep(backoff(attempt))
            continue
        last = (r.status_code, r.text[:300])
        if r.status_code in RETRY_STATUSES:
            time.sleep(backoff(attempt))
            continue
        return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else {})
    return last or (0, "no attempt")


def oa_get(path, params):
    # Hand-built query string: keep ':' and '|' literal inside `filter`.
    parts = []
    for k, v in (params or {}).items():
        if k == "filter":
            parts.append("filter=" + urllib.parse.quote(str(v), safe=":|/"))
        else:
            parts.append("%s=%s" % (k, urllib.parse.quote(str(v), safe="/")))
    if OA_API_KEY:
        parts.append("api_key=" + urllib.parse.quote(OA_API_KEY, safe=""))
    if OA_EMAIL:
        parts.append("mailto=" + urllib.parse.quote(OA_EMAIL, safe=""))
    url = OA_BASE + path + "?" + "&".join(parts)
    last = None
    for attempt in range(MAX_ATTEMPTS):
        _oa_lim.wait()
        try:
            r = requests.get(url, timeout=40)
        except requests.RequestException as e:
            last = ("ERR", str(e))
            time.sleep(backoff(attempt))
            continue
        last = (r.status_code, r.text[:300])
        if r.status_code in RETRY_STATUSES:
            time.sleep(backoff(attempt))
            continue
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, {}
    return last or (0, "no attempt")


# --------------------------------------------------------------------------- #
# OpenAlex abstract reconstruction
# --------------------------------------------------------------------------- #
def oa_abstract(w):
    inv = w.get("abstract_inverted_index")
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #
def s2_resolve(seed, ctx):
    if not S2_API_KEY:
        print("WARNING: S2_API_KEY not set -> shared pool, heavy 429 risk.")
    if seed.get("arxiv_id"):
        cid = "ARXIV:" + str(seed["arxiv_id"])
    elif seed.get("doi"):
        cid = "DOI:" + str(seed["doi"])
    else:
        return None
    st, js = s2_get("/paper/" + cid, {"fields": "title,year,venue,authors,externalIds"})
    ctx.requests += 1
    ctx.bump("S2", st)
    if st == 200 and isinstance(js, dict) and js.get("title"):
        return cid
    if st != 200:
        ctx.err("S2 HTTP %s for %s" % (st, cid))
    return None


def oa_candidates(seed):
    """Collect candidates from several OpenAlex forms; return list of work dicts."""
    out, seen = [], set()

    def add(works):
        for w in works or []:
            wid = (w.get("id") or "").split("/")[-1]
            if wid and wid not in seen:
                seen.add(wid)
                out.append(w)

    title = strip_wild(seed.get("title") or "")
    doi = seed.get("doi")
    if not doi and seed.get("arxiv_id"):
        doi = "10.48550/arxiv." + str(seed["arxiv_id"])

    if doi:
        d = doi.split("doi.org/")[-1].lower()
        for filt in ("doi:https://doi.org/" + d, "doi:" + d):
            st, js = oa_get("/works", {"filter": filt, "per_page": 10})
            add(js.get("results"))

    if title:
        st, js = oa_get("/works", {"filter": "title.search:" + title, "per_page": 25})
        add(js.get("results"))
        st, js = oa_get("/works", {"search": title, "per_page": 25})
        add(js.get("results"))
    return out


def oa_resolve(seed, ctx):
    title = seed.get("title") or ""
    cands = oa_candidates(seed)
    if not cands:
        ctx.unresolved.append("OpenAlex id not resolved: " + title)
        return None, None

    want = None
    doi = seed.get("doi")
    if not doi and seed.get("arxiv_id"):
        doi = "10.48550/arxiv." + str(seed["arxiv_id"])
    if doi:
        want = doi.split("doi.org/")[-1].lower()

    scored = []
    for w in cands:
        sim = title_sim(title, w.get("display_name") or "")
        wdoi = (w.get("doi") or "").split("doi.org/")[-1].lower()
        if want and wdoi == want:
            sim = max(sim, 1.0)
        scored.append((w, sim))
    scored.sort(key=lambda t: (t[1], t[0].get("cited_by_count") or 0), reverse=True)

    best_w, best_sim = scored[0]
    if best_sim < 0.5:
        ctx.unresolved.append("OpenAlex id not resolved (low sim %.2f): %s" % (best_sim, title))
        for w, s in scored[:3]:
            print("     [OA cand] %s sim=%.2f cited_by=%s %s"
                  % ((w.get("id") or "").split("/")[-1], s, w.get("cited_by_count"),
                     short(w.get("display_name") or "", 60)))
        return None, None

    if (best_w.get("cited_by_count") or 0) == 0 and not (best_w.get("referenced_works") or []):
        print("     [OA] WARN: %s -> %s is a fresh record (cited_by=0, refs=0); "
              "OpenAlex has not linked its references yet."
              % (title[:40], (best_w.get("id") or "").split("/")[-1]))
    wid = (best_w.get("id") or "").split("/")[-1]
    diag = {
        "openalex_id": wid,
        "title": best_w.get("display_name"),
        "year": best_w.get("publication_year"),
        "cited_by_count": best_w.get("cited_by_count"),
        "n_referenced_works": len(best_w.get("referenced_works") or []),
        "title_sim": round(best_sim, 3),
    }
    if best_w.get("display_name") and title_sim(title, best_w.get("display_name")) < 0.99:
        ctx.resolved_by_title.append(title + " -> " + str(best_w.get("display_name")))
    return wid, diag


# --------------------------------------------------------------------------- #
# normalisers
# --------------------------------------------------------------------------- #
def norm_row(date_s, db, direction, seed_id, query, **kw):
    row = {k: "" for k in CSV_FIELDS}
    row.update({"date": date_s, "db": db, "direction": direction,
                "seed_id": seed_id, "query": query})
    row.update(kw)
    return row


def norm_s2(p, seed_id, direction, query, date_s):
    ext = p.get("externalIds") or {}
    auth = ", ".join(a.get("name", "") for a in (p.get("authors") or []))
    return norm_row(date_s, "SemanticScholar", direction, seed_id, query,
                    title=p.get("title") or "", year=p.get("year") or "",
                    venue=p.get("venue") or "", authors=auth,
                    arxiv_id=ext.get("ArXiv") or "", doi=ext.get("DOI") or "",
                    s2_paper_id=p.get("paperId") or "")


def norm_oa(w, seed_id, direction, query, date_s, abstract_missing=0):
    auth = ", ".join(a.get("author", {}).get("display_name", "")
                     for a in (w.get("authorships") or []))
    src = ((w.get("primary_location") or {}).get("source") or {})
    wid = (w.get("id") or "").split("/")[-1]
    return norm_row(date_s, "OpenAlex", direction, seed_id, query,
                    title=w.get("display_name") or "", year=w.get("publication_year") or "",
                    venue=src.get("display_name") or "", authors=auth,
                    doi=(w.get("doi") or "").replace("https://doi.org/", ""),
                    openalex_id=wid, abstract_missing=abstract_missing)


# --------------------------------------------------------------------------- #
# keyword matching
# --------------------------------------------------------------------------- #
def flatten_keywords(kw):
    """Accept {group:[k,...]} or [k,...]; return list of (group, keyword)."""
    pairs = []
    if isinstance(kw, dict):
        for g, lst in kw.items():
            for k in (lst or []):
                pairs.append((g, str(k).lower()))
    elif isinstance(kw, list):
        for k in kw:
            pairs.append(("misc", str(k).lower()))
    return pairs


def match_keywords(text, pairs):
    """A phrase matches if ALL of its component words appear anywhere in text."""
    t = (text or "").lower()
    hit, groups = [], set()
    for g, kw in pairs:
        words = [w for w in kw.split() if w]
        if words and all(w in t for w in words):
            hit.append(kw)
            groups.add(g)
    return hit, sorted(groups)


# --------------------------------------------------------------------------- #
# seed snowballing
# --------------------------------------------------------------------------- #
def snowball_seed(seed_id, seed, ctx, date_s, directions, dbs, pairs, min_hits):
    title = seed.get("title") or ""

    # ---- S2 ----
    if "semantic_scholar" in dbs:
        cid = s2_resolve(seed, ctx)
        if not cid:
            print("  [S2] %s -> unresolved" % seed_id)
        else:
            for d in ("forward", "backward"):
                if d not in directions:
                    continue
                endpoint = "/paper/%s/%s" % (cid, "citations" if d == "forward" else "references")
                offset, got, pages = 0, 0, 0
                aborted, reason = False, ""
                while True:
                    st, js = s2_get(endpoint, {"fields": "title,year,venue,authors,externalIds,abstract",
                                               "limit": 100, "offset": offset})
                    ctx.requests += 1
                    ctx.bump("S2", st)
                    if st != 200 or not isinstance(js, dict):
                        aborted, reason = True, "HTTP %s @offset %d" % (st, offset)
                        ctx.err("S2 HTTP %s for %s %s@%d (direction ABORTED, data partial)"
                                % (st, seed_id, d, offset))
                        break
                    data = js.get("data") or []
                    if not data:
                        break
                    pages += 1
                    for item in data:
                        p = item.get("citingPaper") or item.get("citedPaper") or item
                        r = norm_s2(p, seed_id, d, "S2/%s/%s" % (seed_id, d), date_s)
                        text = (r["title"] or "") + " " + (p.get("abstract") or "")
                        r["abstract_missing"] = 0 if p.get("abstract") else 1
                        hit, groups = match_keywords(text, pairs)
                        r["matched_keywords"] = "; ".join(hit)
                        r["keyword_group"] = "; ".join(groups)
                        r["hits"] = len(hit)
                        if len(hit) >= min_hits:
                            r["decision"] = "kept"
                        else:
                            r["decision"] = "check"
                            r["exclude_reason"] = "no keyword hit (kept for auditability)"
                        ctx.rows.append(r)
                        got += 1
                    offset += 100
                    if offset >= 10000:
                        aborted, reason = True, "hit 10k offset cap"
                        break
                ctx.per_seed.setdefault(seed_id, {})["S2:" + d] = got
                ctx.note_complete(seed_id, "S2:" + d, pages, got, aborted, reason)

    # ---- OpenAlex ----
    if "openalex" in dbs:
        wid, diag = oa_resolve(seed, ctx)
        if not wid:
            print("  [OA] %s -> unresolved" % seed_id)
        else:
            print("  [OA] %s -> %s (cited_by=%s, refs=%s, sim=%s)"
                  % (seed_id, wid, diag.get("cited_by_count"),
                     diag.get("n_referenced_works"), diag.get("title_sim")))
            if "forward" in directions:
                cursor, got, pages = "*", 0, 0
                aborted, reason = False, ""
                while True:
                    st, js = oa_get("/works", {"filter": "cites:" + wid, "per_page": 200,
                                               "cursor": cursor})
                    ctx.requests += 1
                    ctx.bump("OA", st)
                    if st != 200:
                        aborted, reason = True, "HTTP %s" % st
                        ctx.err("OA HTTP %s for cites:%s (ABORTED)" % (st, wid))
                        break
                    res = js.get("results") or []
                    if res:
                        pages += 1
                    for w in res:
                        ab = oa_abstract(w)
                        r = norm_oa(w, seed_id, "forward", "OA/%s/forward" % seed_id, date_s,
                                    abstract_missing=0 if ab else 1)
                        hit, groups = match_keywords((r["title"] or "") + " " + ab, pairs)
                        r["matched_keywords"] = "; ".join(hit)
                        r["keyword_group"] = "; ".join(groups)
                        r["hits"] = len(hit)
                        r["decision"] = "kept" if len(hit) >= min_hits else "check"
                        if r["decision"] == "check":
                            r["exclude_reason"] = "no keyword hit (kept for auditability)"
                        ctx.rows.append(r)
                        got += 1
                    cursor = (js.get("meta") or {}).get("next_cursor")
                    if not res or not cursor:
                        break
                ctx.per_seed.setdefault(seed_id, {})["OA:forward"] = got
                ctx.note_complete(seed_id, "OA:forward", pages, got, aborted, reason)

            if "backward" in directions:
                st, js = oa_get("/works", {"filter": "openalex_id:" + wid, "per_page": 1})
                ctx.requests += 1
                ctx.bump("OA", st)
                refs = []
                if st == 200 and (js.get("results") or []):
                    refs = js["results"][0].get("referenced_works") or []
                got, pages = 0, 0
                aborted, reason = False, ""
                for i in range(0, len(refs), 50):
                    ids = "|".join(r.split("/")[-1] for r in refs[i:i + 50])
                    st, js = oa_get("/works", {"filter": "openalex_id:" + ids, "per_page": 100})
                    ctx.requests += 1
                    ctx.bump("OA", st)
                    if st != 200:
                        aborted, reason = True, "HTTP %s on batch %d" % (st, i // 50)
                        ctx.err("OA HTTP %s for referenced batch (ABORTED)" % st)
                        break
                    pages += 1
                    for w in js.get("results") or []:
                        ab = oa_abstract(w)
                        r = norm_oa(w, seed_id, "backward", "OA/%s/backward" % seed_id, date_s,
                                    abstract_missing=0 if ab else 1)
                        hit, groups = match_keywords((r["title"] or "") + " " + ab, pairs)
                        r["matched_keywords"] = "; ".join(hit)
                        r["keyword_group"] = "; ".join(groups)
                        r["hits"] = len(hit)
                        r["decision"] = "kept" if len(hit) >= min_hits else "check"
                        if r["decision"] == "check":
                            r["exclude_reason"] = "no keyword hit (kept for auditability)"
                        ctx.rows.append(r)
                        got += 1
                ctx.per_seed.setdefault(seed_id, {})["OA:backward"] = got
                ctx.note_complete(seed_id, "OA:backward", pages, got, aborted, reason)


# --------------------------------------------------------------------------- #
# keyword sweep (second prong)
# --------------------------------------------------------------------------- #
def keyword_sweep(ctx, date_s, dbs, pairs, min_hits):
    seen = set()
    per_query = {}
    for group, kw in pairs:
        if "semantic_scholar" in dbs:
            st, js = s2_get("/paper/search", {"query": kw, "limit": 20,
                                              "fields": "title,year,venue,authors,externalIds,abstract"})
            ctx.requests += 1
            ctx.bump("S2", st)
            if st == 200 and isinstance(js, dict):
                for p in js.get("data") or []:
                    key = ("S2", p.get("paperId"))
                    if key in seen:
                        continue
                    seen.add(key)
                    per_query["S2:" + kw] = per_query.get("S2:" + kw, 0) + 1
                    r = norm_s2(p, "KW", "keyword", "S2/search:" + kw, date_s)
                    hit, groups = match_keywords((r["title"] or "") + " " + (p.get("abstract") or ""), pairs)
                    r["matched_keywords"] = "; ".join(hit)
                    r["keyword_group"] = group
                    r["hits"] = len(hit)
                    r["decision"] = "kept" if len(hit) >= min_hits else "check"
                    r["abstract_missing"] = 0 if p.get("abstract") else 1
                    ctx.rows.append(r)
            else:
                ctx.err("S2 search '%s' -> HTTP %s" % (kw, st))
        if "openalex" in dbs:
            st, js = oa_get("/works", {"search": strip_wild(kw), "per_page": 20})
            ctx.requests += 1
            ctx.bump("OA", st)
            if st == 200:
                for w in js.get("results") or []:
                    wid = (w.get("id") or "").split("/")[-1]
                    key = ("OA", wid)
                    if key in seen:
                        continue
                    seen.add(key)
                    ab = oa_abstract(w)
                    r = norm_oa(w, "KW", "keyword", "OA/search:" + kw, date_s,
                                abstract_missing=0 if ab else 1)
                    hit, groups = match_keywords((r["title"] or "") + " " + ab, pairs)
                    r["matched_keywords"] = "; ".join(hit)
                    r["keyword_group"] = group
                    r["hits"] = len(hit)
                    r["decision"] = "kept" if len(hit) >= min_hits else "check"
                    ctx.rows.append(r)
    ctx.completeness["KW"] = {"queries": len(pairs), "rows": len(ctx.rows)}


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def write_outputs(ctx, dates, out_dir, cfg_path, args, seeds):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "citation_trace_%s.csv" % dates)
    meta_path = os.path.join(out_dir, "citation_trace_%s.meta.json" % dates)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in ctx.rows:
            w.writerow(r)
    meta = {
        "date": dates,
        "config": cfg_path,
        "direction": args.direction,
        "mode": args.mode,
        "seeds_filter": args.seeds,
        "requests": ctx.requests,
        "status_counter": ctx.status_counter,
        "per_seed": ctx.per_seed,
        "completeness": ctx.completeness,
        "rows_written": len(ctx.rows),
        "unresolved": ctx.unresolved,
        "resolved_by_title": ctx.resolved_by_title,
        "errors": ctx.errors,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("\nRequests: %d | rows: %d" % (ctx.requests, len(ctx.rows)))
    print("Status:  %s" % ctx.status_counter)

    # ---- completeness table (the milestone gate) ----
    print("\nCOMPLETENESS (every non-KW line must say 'ok' for a valid milestone)")
    bad = 0
    for sid in sorted(ctx.completeness):
        if sid == "KW":
            continue
        for direction, info in sorted(ctx.completeness[sid].items()):
            flag = "ABORTED: %s" % info["reason"] if info["aborted"] else "ok"
            if info["aborted"]:
                bad += 1
            print("  %-4s %-12s pages=%-3s items=%-5s %s"
                  % (sid, direction, info["pages"], info["items"], flag))
    if "KW" in ctx.completeness:
        print("  KW   keyword      queries=%s" % ctx.completeness["KW"]["queries"])
    print("  -> %s" % ("ALL COMPLETE" if bad == 0 else "%d direction(s) ABORTED -- re-run those with --seeds" % bad))

    if ctx.errors:
        print("\nErrors (%d):" % len(ctx.errors))
        for e in ctx.errors[:20]:
            print("  - " + e)
    print("\nWrote: %s" % csv_path)
    print("Wrote: %s" % meta_path)


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def do_selftest(seeds, ctx):
    print("== SELFTEST ==")
    print("S2 key set: %s | OpenAlex key set: %s | email: %s"
          % (bool(S2_API_KEY), bool(OA_API_KEY), OA_EMAIL or "(none)"))
    st, js = s2_get("/paper/ARXIV:1803.10122", {"fields": "title"})
    ctx.requests += 1
    ctx.bump("S2", st)
    print("[S2 probe] title=%r statuses={'S2:%s': 1}" % ((js or {}).get("title"), st))
    if st == 429 and S2_API_KEY:
        print("[S2] 429 WITH a key -> key may be inactive, or S2 is throttling under load.")
    if seeds:
        sid = sorted(seeds)[0]
        wid, diag = oa_resolve(seeds[sid], ctx)
        ctx.requests += 1
        ctx.bump("OA", 200 if wid else 0)
        print("[OA probe] %s -> %s" % (sid, diag or "unresolved"))


def do_resolve_only(seeds, ctx):
    print("== RESOLVE ONLY ==")
    for sid in sorted(seeds):
        cid = s2_resolve(seeds[sid], ctx)
        wid, diag = oa_resolve(seeds[sid], ctx)
        print("  %-4s S2=%-28s OA=%s" % (sid, cid or "-", diag or "-"))
    print("Status: %s" % ctx.status_counter)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/seed_papers.yaml")
    ap.add_argument("--out", default="notes/artifacts")
    ap.add_argument("--direction", choices=["forward", "backward", "both"], default="both")
    ap.add_argument("--mode", choices=["snowball", "keyword", "all"], default="all")
    ap.add_argument("--min-hits", type=int, default=1)
    ap.add_argument("--seeds", default="", help="comma list, e.g. S1,S3 (default: all)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--resolve-only", action="store_true")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    seeds = cfg.get("seeds") or {}
    if args.seeds:
        keep = {s.strip() for s in args.seeds.split(",") if s.strip()}
        seeds = {k: v for k, v in seeds.items() if k in keep}
    pairs = flatten_keywords(cfg.get("keywords") or {})
    dbs = (cfg.get("meta") or {}).get("dbs") or ["semantic_scholar", "openalex"]
    dbs = [d for d in ("semantic_scholar", "openalex") if d in dbs]
    dates = date.today().isoformat()

    ctx = Ctx()
    os.makedirs(args.out, exist_ok=True)

    try:
        if args.selftest:
            do_selftest(seeds, ctx)
        elif args.resolve_only:
            do_resolve_only(seeds, ctx)
        elif args.dry_run:
            print("PLAN (%d seeds)  dbs=%s  keywords=%d  s2key=%s oakey=%s"
                  % (len(seeds), dbs, len(pairs), bool(S2_API_KEY), bool(OA_API_KEY)))
            for sid in sorted(seeds):
                s = seeds[sid]
                tr = ",".join(s.get("trace") or [])
                print("  %-4s snowball %-18s %s" % (sid, tr or "(none)", short(s.get("title", ""))))
            return
        else:
            if args.mode in ("snowball", "all"):
                for sid in sorted(seeds):
                    s = seeds[sid]
                    tr = s.get("trace") or []
                    if not tr:
                        continue
                    want = set()
                    if args.direction in ("forward", "both"):
                        want.add("forward")
                    if args.direction in ("backward", "both"):
                        want.add("backward")
                    allow = {("forward" if x == "citations" else "backward") for x in tr}
                    directions = want & allow
                    if not directions:
                        continue
                    snowball_seed(sid, s, ctx, dates, directions, dbs, pairs, args.min_hits)
            if args.mode in ("keyword", "all"):
                keyword_sweep(ctx, dates, dbs, pairs, args.min_hits)
    finally:
        write_outputs(ctx, dates, args.out, args.config, args, seeds)


if __name__ == "__main__":
    main()
