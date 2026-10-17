# Scientific protocol and limits

## Reference and locality

SFL-Lab performs real PyTorch training in a **single-process simulation**. Client fronts and backs own the input, labels and local loss. The server receives detached activations and loss gradients; gradients explicitly cross both cuts. Selected local models start from the global state, are aggregated using sample-weighted FedAvg, and receive the redistributed model. Compliance concerns this message protocol and numerical behavior; it does not imply operating-system isolation or encrypted transport.

ResNet18's eight **residual blocks** define the split units. The stem remains on the front and pooling/classification on the back. Inputs currently use 32×32 RGB preprocessing. The reference is **OS-USFL-inspired**, not a verified reproduction of the original code, preprocessing, data or numerical results. A supplied SeaShips directory enables the image-folder adapter; a paper-reproduction claim additionally needs verified original settings, dataset and full training budget. SeaShips is not bundled.

## Controlled comparisons

A seed determines stratified 70/10/20 train/validation/test indices, client partitions and initial weights. Partitioning supports IID, Dirichlet label skew, sorted label shards and quantity imbalance. Empty partitions are repaired without duplicating samples. Small held-out sets may omit classes. Exact indices, hashes and client label counts are recorded.

Each baseline/candidate pair shares initialization, data, minibatch order and default participation. Noise uses independent random streams. A selection hook intentionally changes client participation or split placement and must be interpreted as that intervention. The reference ignores all candidate settings.

Checkpoint selection uses validation loss. The final held-out test is evaluated once per run after selection. Sweeps choose parameters using validation accuracy. Exploratory suites reuse the research split and cannot establish independent confirmatory evidence; freeze a selected candidate and evaluate a new untouched dataset before publication-level claims. CIFAR-10's official folds are pooled and explicitly repartitioned; its results are not canonical CIFAR-10 benchmark scores.

## Formula and algorithm safety

Expressions use a bounded allowlisted AST interpreter, never `eval`, `exec`, arbitrary Python or a symbolic parser that executes Python. Attributes, imports, subscripting, comprehensions, unknown calls and oversized expressions are rejected. The keyboard includes mathematical notation; only the documented executable subset can be evaluated. Unsupported calculus is reported as an error rather than assigned an invented meaning.

Each hook defines allowed variables and tensor/scalar output contracts. Preflight checks syntax, domains, variables, dimensions, hook placement and mathematical constraints. Runtime rechecks finite outputs, differentiable losses, shapes, positive learning rates and normalized nonnegative aggregation weights. Example-based validation is not a universal mathematical proof.

Algorithm mode permits the required ordered SFL stages and optional clipping/noise operations. It cannot transfer raw inputs or labels to the server. H9 custom accounting formulas are diagnostic and cannot replace the enforced bound. H10 supports offline selection from measured cut sizes; online layer migration is not implemented. H11/H12 change modeled cost equations, not measured physical network or energy outcomes.

## Privacy

L2 clipping bounds the **entire minibatch message tensor** by C. Under replace-one adjacency, sensitivity is conservatively bounded by 2C, including dependence between examples. `sigma` is a noise multiplier: actual Gaussian standard deviation is `sigma * C`. Noise without a verified finite clipping bound, and clipping without positive noise, do not earn a finite guarantee.

The accountant composes `rho(alpha) = alpha * (2C)^2 / (2*(sigma*C)^2)` over all recorded Gaussian releases and converts using `min_alpha[rho(alpha) + log(1/delta)/(alpha-1)]` over its documented orders. There is no subsampling amplification or disjoint-client discount. Unprotected accounted channels yield `epsilon: null` with an explanation, never zero. Per-channel results remain visible.

The bound covers only explicitly accounted split mechanisms under stated assumptions. It does **not** certify unprotected model synchronization, final model releases, side channels, client-level privacy or the complete system. It is not DP-SGD and is not encryption. A finite bound that exceeds the configured budget is a privacy failure even if training succeeds. Budget stopping does not retroactively undo an over-budget release.

Feature and gradient inversion optimize actual pixels against frozen trained reference weights. Clean, noise-only, clipping-only, clipping-plus-noise and candidate message protections share those weights and inputs. MSE, PSNR and SSIM characterize the bounded probes; they are not population attack-success rates. Candidate-trained representations and arbitrary hook effects are not evaluated by a mechanism-isolation probe. Active malicious attacks and unsupported observer capabilities must be reported as unrun, not passed.

## Measurement units and estimates

Accuracy (%), loss, gradients, model changes, tensor shapes, tensor payload bytes and CPU wall time are measured. Absolute accuracy differences are **percentage points**. Each run describes its evaluator traffic and synchronization accounting. Payloads exclude real wire headers, encryption and serialization overhead.

The runtime comparison excludes data download/preparation and one-time optimizer/kernel/split-profile warm-up; their measured durations are stored separately. Warm-up uses a temporary model and zero tensors with the random stream restored. This prevents the first variant from inheriting a dataset-download or lazy-import penalty. Experiment wall time includes setup and supplementary work. CPU scheduling and cache effects still limit fine-grained timing claims.

Network scenarios replay measured payloads with bandwidth, propagation, jitter and expected retransmissions. Retransmission uses `1/(1-p)` expected attempts. They do not delay actual packets, drop training tensors or empirically change accuracy/convergence. Server counts model replica assignment and synchronization in sequential execution; they cannot show real parallel speedup.

Energy uses measured process duration and assumed cycles, frequency, capacitance and radio powers. It is a modeled aggregate estimate, not a wattmeter/GPU measurement or a per-device profile. Model-state bytes are exact; peak hardware memory is not claimed when unmeasured.

## Statistics, acceptance and conclusions

Repeated independent seeds yield means, sample standard deviations and appropriate t confidence intervals. Paired tests match seed identifiers. Missing observations, fewer than three repeats and constant differences are reported without invented p-values or significance. Reported metric tests use Holm adjustment. Small seed counts provide weak evidence even when the numerical test runs.

Acceptance uses the predefined validation-accuracy tolerance and communication/runtime/privacy criteria. Undefined relative changes and comparisons involving unbounded privacy remain undefined. Core SFL failure rejects a candidate; missing evidence makes it inconclusive. Hypothesis status assesses the entered numeric criteria, not an unrestricted natural-language or causal claim. Passing observed means does not establish equivalence, statistical significance or general superiority.

Pareto points require finite observations for each chosen objective. Conclusions remain restricted to the tested dataset, model, splits, seeds, training budget and simulated conditions. No cross-dataset generalization, original-paper result or full-system security is inferred.
