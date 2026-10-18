# Changelog

All dates are local (UTC+8). Retractions are numbered and recorded in
`paper/callback_memo.md`; each entry links to the note that decided it.

| date | change |
| --- | --- |
| 2026-10-18 | README restructured around the action-channel study: two-part title, a Highlights list, the dose–response finding moved ahead of the reproduction results, a four-line Quickstart, and a Docs index. Install/run details moved into `REPRODUCE.md`; this changelog moved here. |
| 2026-10-17 | **Documentation parity fixes.** Install path unified to `pip install -r requirements.txt` (the clean-container verified path); the "system-level swig" claim removed (`box2d-py` resolves to a wheel); checkpoint size corrected to ≈ 429 MB / 87 files (`runs/**/*.pt` ignored, `log.jsonl` tracked across 36 files); `REPRODUCE.md` Tier-1/2 command blocks deduplicated; torch pinned to `2.5.1` (the `+cu121` suffix was local-only). **Clean-container verification** (Ubuntu 22.04, nothing preinstalled): every stage re-run at V-full / otherwise-smoke scale, Tier-1 table recomputation exit 0. |
| 2026-10-09 | **CarRacing C convergence corrected — not converged (retraction #5).** `notes/Note6.md` §2/§7 said "saturated / converges"; generation 99 still set a new best (12,800 vs 1.84 M rollouts, 1/144), so the run is **not converged** — the README CarRacing section is authoritative. Ledger: `paper/callback_memo.md` |
| 2026-10-01 | Action-channel study frozen (`notes/Note11.md`), audit added (`scripts/mdn/n18_audit.py`), repository history slimmed |
| 2026-09-21 | C complete (619.65, +94.50 paired); §3.4 dream; score distribution analysed (Notes 6–7) |
| 2026-09-20 | C module added; V-only arm (525.15); R²-over-persistence retracted (Note5) |
| 2026-09-19 | M trained and closed; `--norm-min-std` retracted (Note4) |
| 2026-09-17 | V accepted at R² 0.916 after three I/O and shape fixes (Note3) |
| 2026-09-16 | Data investigation corrected Note1's root cause (Note2) |
