# Kickoff prompt for a fresh conversation

Copy the block below verbatim into a new Claude Code session at the
`/Users/markusreichel/PhD/tudatpy` working directory.

---

I need you to implement constrained multi-arc orbit determination
continuity in tudat/tudatpy. The complete specification — math, file
paths, line numbers, signatures, test list — is in
`multi_arc_implementation.md` at the repo root.

Please:

1. Read `multi_arc_implementation.md` end to end before writing any code.
   It is self-contained: do not consult the earlier draft plans
   `claude_multi_arc.md` / `codex_multi_arc.md` unless I ask you to.

2. Verify each cited file:line anchor in §3 and §10 of the spec before
   touching it — the spec is current as of this branch
   (`constrained-multi-arc-orbit-estimation`) but the tree may have
   moved.

3. Follow the implementation order in §8 strictly:
   - Start with the per-arc-index STM accessor (§4.1) and its unit test
     (test 7 in §6) before anything else. Compile and run that test
     green before moving on.
   - Then the LSQ optional additions (§4.2) with the empty-no-op test
     (test 11). Compile and run green.
   - Then settings (§4.3), then the assembly module (§4.4 / §4.5), then
     the wiring (§4.6 / §4.7), then input/output (§4.8), then Python
     (§4.10). Run the relevant tests from §6 at each step.

4. For every modification, prefer the minimal change that satisfies the
   spec. Do not refactor surrounding code, do not add unrelated
   abstractions, do not introduce backwards-compatibility shims.

5. Where the spec lists multiple options or asks me to choose, ask me
   first — do not pick silently.

6. Convention reminders (call these out if you find conflicting code):
   - Cost: `Q_d = (1/m_d) · (1/μ) · Σ d^T C d`. Larger μ weakens.
   - Residual sign: continuity residual is `−d`, so
     `g_constraint += −D^T W_d d`, `H_constraint += D^T W_d D`.
   - `D = M_right − M_left` using the new per-arc-index STM accessor.
   - Column-normalize `D` with Tudat's existing `normalizationTerms`
     **before** computing `H_constraint` and `g_constraint`. `d` itself
     stays in physical units.

7. End-of-task deliverables:
   - All §6 unit tests passing.
   - A short summary listing every file you touched, with one-line per
     file.
   - One-line summary of the §7 Rosetta verification status (run if
     SPICE kernels are locally available; otherwise note "deferred to
     user").

Background context if helpful: the feature follows Cicalò et al. (2021)
§3.3 Eq. (27)–(28) — "constrained multi-arc strategy" used in
BepiColombo OD. The motivating Rosetta case constrains spacecraft
position at OCM epochs while leaving velocity free to jump (Δv is not
estimated). See §1 of the spec for full background.
