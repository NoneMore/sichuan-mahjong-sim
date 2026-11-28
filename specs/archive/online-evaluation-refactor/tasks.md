# Online evaluation refactor — Tasks

[spec.md](spec.md) owns acceptance; [design.md](design.md) owns architecture. [Verification](verification.md) records the results.

## M0 — Baseline and interoperability

- [x] **M0.1 · Capture regression baseline.** Use the fixed pre-change revision; pin config/seed/checkpoint identities; capture existing 2N2T traces, terminal records, counts, summary and batch/forced behavior. Identify existing native teacher replay and early-Hu fixtures; record small matched timing workload.
- [x] **M0.2 · Widen compare eligibility.** Accept `best`/`policy`, reject `last`, preserve checkpoint loader/schema and compare metadata. Update CLI/help/tests, retaining role-negative and corruption cases; smoke ordinary v3 policy export with one legal validation seed and six placements. **Independent of kernel extraction.**

## M1 — Neutral execution kernel

- [x] **M1.1 · Public plan and selector contract.** Define stable game IDs, four policy assignments, registry, requests/choices and neutral terminal results. Validate inputs before Env creation and neutral terminal facts independently of 2N2T restrictions. Test Plan and identity requirements in spec, including repeated seeds, duplicate IDs, invalid registry and seat assignments.
- [x] **M1.2 · Extract synchronous loop.** Use one Env per active game, deterministic refill/group/chunk order, public native teacher registration and shared cache, forced bypass, round prevalidation and contextual failures. Implement per-seat/policy counts and selector/teacher diagnostics. Test wrong-policy routing, native-ID collisions, group/chunk order with forced rows and refill, batch bounds, invalid results before any round step, limits, terminal facts and plan-order results under out-of-order completion. **Depends: M0.1, M1.1.**
- [x] **M1.3 · Observer contract.** Expose owned pre-action request, accepted choice and opaque metadata; notify before terminal/refill, never for failed steps. Test owned snapshots, identity/metadata alignment, forced events, final decision before terminal notification, immediate callback failure and no feature retention by default. **Depends: M1.2.**
- [x] **M1.4 · Neural selectors.** Reuse frozen greedy inference and legal-mask sampling with explicit generator; support multiple independent models and configured selectors. Test canonical ties, padding, nonfinite valid logits and deterministic fixed-config sampling. **Depends: M1.1.**

## M2 — Adapt existing comparison

- [x] **M2.1 · Legacy adapter.** Route existing scheduler API through neutral kernel, project trace/count/timing/results to old shapes; retain test seam and existing records protocol. Delete duplicated execution loop. Verify M0.1 fixtures and existing online tests, including legacy inference accounting and timing boundaries. **Depends: M0.1, M1.2–M1.4.**
- [x] **M2.2 · CLI and persistence regression.** Verify best/policy eligibility, v3 policy compare smoke, last rejection before run publication, split checks, six placements, v1 records, persisted summary, costs, incomplete failure behavior and old-record readability. **Depends: M0.2, M2.1.**

## M3 — Integration and handoff

- [x] **M3.1 · Native smoke and contracts.** Complete 2N2T, 4N, 4T and two-neural 2A2B through the shared kernel; match native teacher replay and early-Hu/final settlement fixture; test sampling replay, unchanged sampling generator state for forced bypass and a caller-owned metadata observer without RL dependency. Run existing Rust/Python and online checks per [development](../../../docs/development.md#running). **Depends: M2.1.**
- [x] **M3.2 · Diagnostic timing and CI.** Repeat M0.1 workload under matched conditions; record timings/throughput and findings, investigate apparent regressions before handoff, without a numerical performance gate. Include new contract/smoke tests in existing online CPU CI. **Depends: M2.2, M3.1.**
- [x] **M3.3 · Public API and docs.** Document execution usage, identity, ordinary selectors, native teacher registration with shared cache, observer lifetime, sampling repeatability, error/partial-effect semantics and compare role change in owning package/architecture/contracts. Replace wording implying compare requires offline-selected best or forbids online model selection. Run docs checks. Archive after acceptance; downstream RL imports only documented public interfaces. **Depends: M3.2.**
