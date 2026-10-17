# Verified execution and release scope

Validation uses the real model, serial worker, SQLite store, HTTP API and browser. No experiment measurements are mocked in the application or environment verifier. The tests use controlled fixtures only for testing decision arithmetic and invalid inputs.

## Checks

| Check | Executed scope |
| --- | --- |
| Automated Python tests | 275 passing tests: unsafe formula rejection, constraints, all executable templates and H1–H14 in real training, split/unsplit gradient equivalence, five ResNet18 splits, partitions, privacy composition, paired statistics, budget decisions, real 2D sweep, ablations, API queue/cancellation/persistence and reports |
| Setup | Locked Python install, frozen npm install, editable package installation, frontend production build and dependency consistency; repeated successfully |
| Complete API evaluation | Tiny CNN, 100 Digits samples, two clients, one round, three paired seeds; active activation/gradient protection and FedProx; all seven supplementary suites |
| Additional model check | ResNet18, 40 samples, two clients, one round, paired reference/candidate |
| Learning check | ResNet18, 1,200 Digits samples, three clients, eight rounds, Adam 0.001, batch 32, seed 11 |
| Browser | Actual UI-launched paired training; safe formula/algorithm controls; all 13 result tabs; real network heatmap and reconstruction images; PNG/SVG/PDF downloads; raw evidence; light/dark themes; cancellation; 390px mobile navigation and overflow checks |

The learning run selected round 5 by validation loss: **95.0% validation accuracy and 91.67% held-out test accuracy**. This is a functional, single-seed Digits result. It does not measure a candidate improvement, SeaShips recognition, generalization or a statistically reliable population score. The complete evaluator's short budgets test execution, not convergence.

Generated evidence is stored locally in `artifacts/verification.json`, `artifacts/verification-reports/`, `artifacts/learning-validation.json`, and `artifacts/browser/`. JSON exports contain exact configuration, split indices/hashes, model fingerprints, seeds, software/Git provenance, logs, measured counters and limitations. Generated datasets and reports are ignored by Git; CI uploads verifier evidence separately.

## Boundaries of this release

| Capability | Status and interpretation |
| --- | --- |
| U-shaped training and fixed/candidate comparison | Implemented and exercised, in one CPU process with logical client/server boundaries |
| Dataset adapters | Digits and local image-folder adapter exercised; SeaShips is not supplied. CIFAR-10 download is optional and not part of offline readiness |
| Formula/algorithm execution | Bounded operation language; mathematical notation outside that executable subset is rejected |
| Privacy | Conservative Gaussian RDP for explicitly bounded/noised split releases; no model/synchronization/end-to-end DP certification |
| Networks and energy | Model estimates using measured tensor sizes/durations; no real network emulation, packet-drop training failures or physical power measurement |
| Scalability | Functional checks use clients 2/3 and server replicas 1/2. Larger configurable counts remain experiments to execute, not demonstrated distributed scaling |
| Splits | All five static profiles and offline cost selection; online layer migration and adaptive noise scheduling are not implemented |
| Attacks | Real feature and split-gradient pixel optimization on three training examples using one trained frozen reference; passive probes, not active malicious attacks or a population success rate |
| Statistics | Seed-paired estimates, t tests when valid, Holm adjustment, explicit missing/constant evidence and validation-based selection; exploratory suites need independent confirmation |
| Reporting | PDF/CSV/JSON/Markdown plus chart PNG/SVG; complete raw evidence in JSON; PDF includes performance and validation curves rather than every possible dashboard chart |
| Persistence and deployment | SQLite history and cancellation work. Persistent model resume, multi-process training, production authentication and deployment are outside this local research release |

Re-run the commands in the README for the exact machine and chosen data before using a result. The [methodology](METHODOLOGY.md) explains units, threat models, privacy assumptions, selection rules and scientific limits.
