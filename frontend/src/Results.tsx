import { useState } from "react";
import {
  ArrowUpRight,
  ArrowRight,
  ShieldCheck,
  FlaskConical,
  CircleCheck,
  CircleAlert,
  CircleHelp,
  FileDown,
  Database,
  Eye,
  Check,
  Layers3,
  Cpu,
  Network,
  Info,
} from "lucide-react";
import { Chart, Empty, lineOption } from "./Charts";
import {
  bytes,
  colors,
  fmt,
  metric,
  metricNames,
  metricUnits,
  numeric,
  title,
} from "./types";
import type { Data, Experiment } from "./types";
export function Badge({
  status,
  children,
}: {
  status?: string;
  children?: React.ReactNode;
}) {
  const s = String(status || "pending").toLowerCase();
  const tone = /^(pass|completed|supported|safe|accept|valid)$/.test(s)
    ? "good"
    : /fail|reject|not_supported|invalid/.test(s)
      ? "bad"
      : /warning|running|partial|condition/.test(s)
        ? "warn"
        : "muted";
  return (
    <span className={`badge ${tone}`}>
      <i />
      {children || title(status || "Pending")}
    </span>
  );
}
export function Checks({ checks }: { checks: Data[] }) {
  return (
    <div className="checks">
      {checks.map((c, i) => (
        <div className="check-row" key={i}>
          <span
            className={
              "check-symbol " +
              (/^(pass|safe|supported|completed|valid)$/i.test(c.status)
                ? "good"
                : /fail|reject|invalid/i.test(c.status)
                  ? "bad"
                  : "muted")
            }
          >
            {/^(pass|safe|completed|valid)$/i.test(c.status) ? (
              <CircleCheck size={16} />
            ) : /fail|reject|invalid/i.test(c.status) ? (
              <CircleAlert size={16} />
            ) : (
              <CircleHelp size={16} />
            )}
          </span>
          <div>
            <strong>{c.name || c.label}</strong>
            <p>{c.detail || c.reason || ""}</p>
          </div>
          <Badge status={c.status} />
        </div>
      ))}
    </div>
  );
}
function Limitations({ items }: { items: any[] }) {
  return items?.length ? (
    <div className="note">
      <Info size={16} />
      <div>
        <strong>Scope & interpretation</strong>
        <ul>
          {items.map((s, i) => (
            <li key={i}>{typeof s === "string" ? s : JSON.stringify(s)}</li>
          ))}
        </ul>
      </div>
    </div>
  ) : null;
}
function MetricCard({
  label,
  value,
  unit,
  delta,
  kind = "measured",
  tip,
}: Data) {
  return (
    <div className="metric-card panel">
      <div className="metric-label" title={tip}>
        {label}
        <Info size={12} />
      </div>
      <div className="metric-value">
        {numeric(value) ? fmt(value) : "—"}
        <span>{unit}</span>
      </div>
      <div className="metric-foot">
        <span className={`data-kind ${kind === "estimate" ? "estimate" : ""}`}>
          {kind === "estimate" ? "ESTIMATE" : "MEASURED"}
        </span>
        {delta && numeric(delta.absolute) && (
          <span title="Candidate minus baseline">
            {delta.absolute > 0 ? "+" : ""}
            {fmt(delta.absolute)} {delta.unit || unit}
          </span>
        )}
      </div>
    </div>
  );
}
export function SplitDiagram({
  split = [2, 4, 2],
  active = false,
}: {
  split?: number[];
  active?: boolean;
}) {
  const front = split[0],
    middle = split[1];
  return (
    <div className={`split-diagram ${active ? "is-active" : ""}`}>
      <div className="split-stages">
        <span>CLIENT FRONT</span>
        <span>SERVER MIDDLE</span>
        <span>CLIENT BACK</span>
      </div>
      <div className="layers">
        {Array.from({ length: 8 }, (_, i) => (
          <div
            key={i}
            className={`layer ${i < front ? "front" : i < front + middle ? "middle" : "back"}`}
          >
            <Layers3 size={15} />
            <strong>L{i + 1}</strong>
            <span>
              {i === front - 1 || i === front + middle - 1 ? "CUT" : "BLOCK"}
            </span>
          </div>
        ))}
      </div>
      <div className="flow-caption">
        <span>
          <Database size={13} /> Private data Xₖ
        </span>
        <span>
          Activations Aₖ <ArrowRight size={12} />
        </span>
        <span>
          Representation Zₖ <ArrowRight size={12} />
        </span>
        <span>Local labels Yₖ</span>
      </div>
      <div className="backward-line">
        <span>← Split backward gradients</span>
        <span>Local loss · ℓ(ŷₖ, Yₖ)</span>
      </div>
    </div>
  );
}
function SuitePanel({
  suite,
  name,
  dark,
  parameter = "label",
}: {
  suite?: Data;
  name: string;
  dark: boolean;
  parameter?: string;
}) {
  const [m, setM] = useState("val_accuracy");
  const rows = Array.isArray(suite?.rows) ? suite!.rows : [];
  const available = Object.keys(metricNames).filter((k) =>
    rows.some((r: Data) => numeric(r.metrics?.[k])),
  );
  const chosen = available.includes(m) ? m : available[0];
  const data = rows.map((r: Data) => r.metrics?.[chosen]);
  const option =
    rows.length && chosen
      ? {
          xAxis: {
            type: "category",
            data: rows.map((r: Data) =>
              String(
                r[parameter] ??
                  r.label ??
                  r.scenario ??
                  r.value ??
                  r.clients ??
                  "",
              ),
            ),
            axisLabel: { rotate: rows.length > 8 ? 30 : 0 },
          },
          yAxis: { type: "value", name: metricUnits[chosen] },
          series: [
            {
              name: metricNames[chosen],
              type: "bar",
              barMaxWidth: 42,
              itemStyle: {
                color: colors.candidate,
                borderRadius: [4, 4, 0, 0],
              },
              data,
            },
          ],
        }
      : null;
  return (
    <>
      <div className="result-toolbar">
        <div>
          <h3>{name}</h3>
          <Badge status={suite?.status || "not_run"} />
        </div>
        {available.length > 0 && (
          <select
            aria-label={`${name} metric`}
            value={chosen}
            onChange={(e) => setM(e.target.value)}
          >
            {available.map((k) => (
              <option key={k} value={k}>
                {metricNames[k]}
              </option>
            ))}
          </select>
        )}
      </div>
      {!rows.length ? (
        <Empty
          title={`${name} has not produced results`}
          detail={
            suite?.reason ||
            suite?.detail ||
            "Enable this evaluation in Experiment Setup and run the full evaluation."
          }
        />
      ) : (
        <>
          <Chart
            title={`${metricNames[chosen] || name} across conditions`}
            subtitle="Each point comes from an executed condition. Selection uses validation performance."
            dark={dark}
            option={option}
          />
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Condition</th>
                  <th>Validation</th>
                  <th>Test</th>
                  <th>Runtime</th>
                  <th>Communication</th>
                  <th>Privacy ε</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r: Data, i: number) => (
                  <tr key={i}>
                    <td>
                      {String(
                        r.label ??
                          r.scenario ??
                          r.value ??
                          r[parameter] ??
                          i + 1,
                      )}
                    </td>
                    <td>{metric(r.metrics?.val_accuracy, "val_accuracy")}</td>
                    <td>{metric(r.metrics?.test_accuracy, "test_accuracy")}</td>
                    <td>
                      {metric(r.metrics?.runtime_seconds, "runtime_seconds")}
                    </td>
                    <td>{bytes(r.metrics?.communication_bytes)}</td>
                    <td>{fmt(r.metrics?.epsilon)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      <Limitations items={suite?.limitations || []} />
    </>
  );
}
export default function Results({
  experiment,
  tab,
  dark,
  onEvidence,
}: {
  experiment?: Experiment;
  tab: string;
  dark: boolean;
  onEvidence: () => void;
}) {
  const [paretoMetric, setParetoMetric] = useState("communication_bytes");
  const [networkVariant, setNetworkVariant] = useState("candidate");
  const [networkMetric, setNetworkMetric] = useState("latency_seconds");
  const result = experiment?.result,
    b: Data[] = result?.baseline || [],
    c: Data[] = result?.candidate || [],
    comp: Data = result?.comparison || {},
    suites: Data = result?.suites || {};
  const aggregate = (variant: string, key: string) => {
    const measured = comp[variant]?.[key];
    if (numeric(measured?.mean)) return measured.mean;
    const runs = variant === "baseline" ? b : c;
    const vals = runs.map((r) => r.metrics?.[key]).filter(numeric);
    if (key === "epsilon" && vals.length !== runs.length) return null;
    return vals.length ? vals.reduce((a, v) => a + v, 0) / vals.length : null;
  };
  const latest = c[c.length - 1] || b[b.length - 1];
  if (!result)
    return (
      <div className="results-empty panel">
        <div className="empty-grid" />
        <span className="empty-orbit">
          <FlaskConical size={31} />
        </span>
        <span className="eyebrow">EVIDENCE STARTS WITH AN EXPERIMENT</span>
        <h2>Your next hypothesis, tested.</h2>
        <p>
          Define a candidate, keep the baseline controlled, and let measured
          results tell the story. This workspace fills as your experiments
          complete.
        </p>
        <div className="empty-steps">
          <span>
            <Check size={14} /> Define
          </span>
          <ArrowRight size={13} />
          <span>
            <ShieldCheck size={14} /> Validate
          </span>
          <ArrowRight size={13} />
          <span>
            <FlaskConical size={14} /> Experiment
          </span>
          <ArrowRight size={13} />
          <span>
            <ChartIcon /> Evaluate
          </span>
        </div>
      </div>
    );
  const complianceByName = new Map<string, Data>();
  for (const run of [...b, ...c])
    for (const check of run.compliance || []) {
      const previous = complianceByName.get(check.name);
      const rank = (status: string) =>
        /fail/i.test(status) ? 2 : /^pass$/i.test(status) ? 0 : 1;
      if (!previous || rank(check.status) > rank(previous.status))
        complianceByName.set(check.name, {
          ...check,
          detail: `${title(run.variant)} · seed ${run.seed}: ${check.detail}`,
        });
    }
  const compliance = [...complianceByName.values()];
  const comparisonNotice =
    !b.length || !c.length ? (
      <div className="note">
        <Info size={16} />
        <span>
          Only {b.length ? "baseline" : "candidate"} measurements are available.
          Run a paired comparison to evaluate candidate effects.
        </span>
      </div>
    ) : null;
  if (tab === "Overview")
    return (
      <>
        {comparisonNotice}
        <div className="metric-grid">
          <MetricCard
            label="Test accuracy"
            value={
              aggregate("candidate", "test_accuracy") ??
              aggregate("baseline", "test_accuracy")
            }
            unit="%"
            delta={comp.deltas?.test_accuracy}
            tip="Final test accuracy for the checkpoint selected using validation data."
          />
          <MetricCard
            label="Validation accuracy"
            value={
              aggregate("candidate", "val_accuracy") ??
              aggregate("baseline", "val_accuracy")
            }
            unit="%"
            delta={comp.deltas?.val_accuracy}
            tip="Validation performance guides model selection; the test set is held out."
          />
          <MetricCard
            label="Execution time"
            value={
              aggregate("candidate", "runtime_seconds") ??
              aggregate("baseline", "runtime_seconds")
            }
            unit="s"
            delta={comp.deltas?.runtime_seconds}
            tip="Measured wall time of local training in this environment."
          />
          <MetricCard
            label="Energy consumption"
            value={
              aggregate("candidate", "energy_joules") ??
              aggregate("baseline", "energy_joules")
            }
            unit="J"
            delta={comp.deltas?.energy_joules}
            kind="estimate"
            tip="Model estimate from compute time, device parameters, and network transfer."
          />
        </div>
        <div className="chart-grid">
          <Chart
            title="Validation accuracy"
            subtitle="Baseline vs candidate · mean across available seeds"
            option={lineOption(b, c, "val_accuracy", "Accuracy (%)", true)}
            dark={dark}
          />
          <Chart
            title="Validation loss"
            subtitle="Checkpoints are selected using validation performance"
            option={lineOption(b, c, "val_loss", "Loss")}
            dark={dark}
          />
        </div>
        <section className="recommendation panel">
          <div className="recommendation-icon">
            <FlaskConical size={24} />
          </div>
          <div>
            <span className="eyebrow">SCIENTIFIC CONCLUSION</span>
            <h3>{comp.recommendation?.decision || "Comparison pending"}</h3>
            <p>
              {comp.recommendation?.reasons?.join(" ") ||
                "A paired baseline and candidate comparison is required before drawing a conclusion."}
            </p>
            <Badge status={comp.hypothesis_status || "inconclusive"}>
              Hypothesis: {title(comp.hypothesis_status || "inconclusive")}
            </Badge>
          </div>
          <button className="button secondary" onClick={onEvidence}>
            <Eye size={14} /> View evidence <ArrowUpRight size={13} />
          </button>
        </section>
        <div className="chart-grid">
          <section className="panel">
            <div className="section-heading">
              <h3>Acceptance criteria</h3>
              <span className="tiny-label">PREDEFINED</span>
            </div>
            {comp.criteria?.length ? (
              <Checks checks={comp.criteria} />
            ) : (
              <Empty title="Paired comparison required" />
            )}
          </section>
          <section className="panel">
            <div className="section-heading">
              <h3>Core SFL compliance</h3>
              <ShieldCheck size={17} />
            </div>
            {compliance.length ? (
              <Checks checks={compliance} />
            ) : (
              <Empty title="Compliance not yet evaluated" />
            )}
          </section>
        </div>
        <Limitations
          items={[...(result.limitations || []), ...(comp.limitations || [])]}
        />
      </>
    );
  if (tab === "Architecture")
    return (
      <>
        <section className="panel">
          <div className="section-heading">
            <div>
              <h3>U-shaped split learning architecture</h3>
              <p>
                Logical clients · detached tensor exchange at both split
                boundaries
              </p>
            </div>
            <span className="badge muted">SIMULATION</span>
          </div>
          <SplitDiagram split={experiment?.config.split} />
          <div className="architecture-cards">
            <div>
              <Cpu size={19} />
              <strong>{experiment?.config.clients} clients</strong>
              <p>Local data, labels, front and back blocks</p>
            </div>
            <div>
              <Network size={19} />
              <strong>{experiment?.config.servers} server replicas</strong>
              <p>Simulated assignment; no distributed speedup claim</p>
            </div>
            <div>
              <Layers3 size={19} />
              <strong>{title(experiment?.config.model)}</strong>
              <p>Sample-weighted federated baseline</p>
            </div>
          </div>
        </section>
        <section className="panel">
          <div className="section-heading">
            <h3>Observed architectural invariants</h3>
          </div>
          <Checks checks={compliance} />
        </section>
        <SuitePanel
          suite={suites.split_profile}
          name="Split profiling"
          dark={dark}
        />
        <Limitations
          items={[
            "Logical client boundaries are simulated in one process. They do not provide operating-system or cryptographic isolation.",
            "The reference architecture is inspired by OS-USFL. Numerical reproduction requires verified original data and settings.",
          ]}
        />
      </>
    );
  if (tab === "Learning")
    return (
      <>
        {comparisonNotice}
        <div className="chart-grid">
          <Chart
            title="Training accuracy"
            subtitle="Mean per round · measured (%)"
            option={lineOption(b, c, "train_accuracy", "Accuracy (%)", true)}
            dark={dark}
          />
          <Chart
            title="Validation accuracy"
            subtitle="Mean per round · measured (%)"
            option={lineOption(b, c, "val_accuracy", "Accuracy (%)", true)}
            dark={dark}
          />
          <Chart
            title="Training loss"
            subtitle="Mean per round · measured"
            option={lineOption(b, c, "train_loss", "Loss")}
            dark={dark}
          />
          <Chart
            title="Validation loss"
            subtitle="The test set is excluded from checkpoint selection"
            option={lineOption(b, c, "val_loss", "Loss")}
            dark={dark}
          />
        </div>
        <section className="panel">
          <div className="section-heading">
            <h3>Final checkpoint evidence</h3>
          </div>
          <RunTable baseline={b} candidate={c} />
        </section>
      </>
    );
  if (tab === "Privacy") {
    const privacy = latest?.privacy || {};
    const eps = latest?.metrics?.epsilon;
    const max = experiment?.config.privacy.epsilon_max || 10;
    return (
      <>
        <div className="privacy-summary panel">
          <div>
            <span className="eyebrow">CUMULATIVE TRANSCRIPT ACCOUNTING</span>
            <h2>
              {fmt(eps)} <span>/ {fmt(max)} ε</span>
            </h2>
            <p title="Lower epsilon indicates stronger privacy under this accountant’s stated assumptions.">
              ε is scoped to accounted tensor releases.
            </p>
          </div>
          <div>
            <Badge
              status={
                privacy.status ||
                (!numeric(eps)
                  ? "not_accounted"
                  : eps >= max
                    ? "fail"
                    : eps / max >= 0.8
                      ? "warning"
                      : "safe")
              }
            />
            <div className="budget-track">
              <div
                style={{
                  width: `${numeric(eps) ? Math.min(100, (100 * eps) / max) : 0}%`,
                }}
              />
            </div>
            <small>
              δ = {experiment?.config.privacy.delta} ·{" "}
              {title(experiment?.config.privacy.threat_model)}
            </small>
          </div>
        </div>
        <Chart
          title="Privacy budget by released message"
          subtitle="Cumulative ε including training and evaluation releases · mean of available seeds · missing bounds are not zero"
          option={
            [...b, ...c].some((r) =>
              (r.privacy?.history || []).some((h: Data) => numeric(h.epsilon)),
            )
              ? {
                  xAxis: {
                    type: "value",
                    name: "Release count",
                    minInterval: 1,
                  },
                  yAxis: { type: "value", name: "ε" },
                  series: [
                    ["Baseline", b],
                    ["Candidate", c],
                  ].map(([name, runs]) => {
                    const grouped = new Map<number, number[]>();
                    for (const run of runs as Data[])
                      for (const h of run.privacy?.history || [])
                        if (numeric(h.epsilon))
                          grouped.set(h.release_count, [
                            ...(grouped.get(h.release_count) || []),
                            h.epsilon,
                          ]);
                    return {
                      name,
                      type: "line",
                      symbolSize: 4,
                      data: [...grouped]
                        .sort((a, b) => a[0] - b[0])
                        .map(([count, values]) => [
                          count,
                          values.reduce((a, v) => a + v, 0) / values.length,
                        ]),
                      ...(name === "Candidate"
                        ? {
                            markLine: {
                              symbol: "none",
                              data: [
                                {
                                  yAxis: max,
                                  name: "Maximum ε",
                                  lineStyle: { color: colors.red },
                                },
                                {
                                  yAxis:
                                    max *
                                    (experiment?.config.privacy
                                      .warning_fraction || 0.8),
                                  name: "Warning",
                                  lineStyle: { color: colors.amber },
                                },
                              ],
                            },
                          }
                        : {}),
                    };
                  }),
                }
              : null
          }
          dark={dark}
        />
        <div className="chart-grid">
          <section className="panel">
            <div className="section-heading">
              <h3>Configured protection</h3>
            </div>
            <dl className="definition-list">
              {Object.entries(experiment?.config.candidate || {})
                .filter(([k]) => /clip|noise|mu/.test(k))
                .map(([k, v]) => (
                  <div key={k}>
                    <dt>{title(k)}</dt>
                    <dd>{v == null ? "Disabled" : String(v)}</dd>
                  </div>
                ))}
            </dl>
            <p className="caption">
              Formula parameters may additionally configure protection at the
              selected hook. Open evidence for the complete effective
              configuration.
            </p>
          </section>
          <section className="panel">
            <div className="section-heading">
              <h3>Accountant evidence</h3>
            </div>
            <pre className="json-preview">
              {JSON.stringify(privacy, null, 2)}
            </pre>
          </section>
        </div>
        <Limitations
          items={[
            "This accountant describes protected tensor transcripts under its sensitivity and composition assumptions. It does not certify end-to-end privacy of the full model, synchronization updates, or complete system.",
            "Clipping without calibrated noise does not establish differential privacy. Missing ε is not zero privacy cost.",
          ]}
        />
      </>
    );
  }
  if (tab === "Federation") {
    const clients = latest?.clients || [];
    return (
      <>
        <div className="chart-grid">
          <Chart
            title="Client accuracy"
            subtitle="Local client evaluation · measured (%)"
            dark={dark}
            option={
              clients.length
                ? {
                    xAxis: {
                      type: "category",
                      data: clients.map((r: Data) => `Client ${r.id}`),
                    },
                    yAxis: {
                      type: "value",
                      name: "Accuracy (%)",
                      min: 0,
                      max: 100,
                    },
                    series: [
                      {
                        name: "Accuracy",
                        type: "bar",
                        data: clients.map((r: Data) => r.accuracy),
                        itemStyle: { color: colors.candidate },
                        barMaxWidth: 40,
                      },
                    ],
                  }
                : null
            }
          />
          <Chart
            title="Client drift"
            subtitle="L2 distance from local to global parameters"
            dark={dark}
            option={
              clients.length
                ? {
                    xAxis: {
                      type: "category",
                      data: clients.map((r: Data) => `Client ${r.id}`),
                    },
                    yAxis: { type: "value", name: "L2 drift" },
                    series: [
                      {
                        name: "Drift",
                        type: "bar",
                        data: clients.map((r: Data) => r.drift),
                        itemStyle: { color: colors.baseline },
                        barMaxWidth: 40,
                      },
                    ],
                  }
                : null
            }
          />
        </div>
        <div className="table-wrap panel">
          <table>
            <thead>
              <tr>
                <th>Client</th>
                <th>Training samples</th>
                <th>Accuracy</th>
                <th>Loss</th>
                <th>Drift</th>
              </tr>
            </thead>
            <tbody>
              {clients.map((r: Data) => (
                <tr key={r.id}>
                  <td>Client {r.id}</td>
                  <td>{r.samples}</td>
                  <td>{fmt(r.accuracy)}%</td>
                  <td>{fmt(r.loss, 4)}</td>
                  <td>{fmt(r.drift, 4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <SuitePanel
          suite={suites.non_iid}
          name="Non-IID robustness"
          dark={dark}
        />
      </>
    );
  }
  if (tab === "Communication") {
    const comm = latest?.communication || {};
    const entries = Object.entries(comm).filter(([, v]) => numeric(v));
    const stageNames: Data = {
      activation: "Front → Server",
      intermediate: "Server → Back",
      gradient_back: "Back → Server",
      gradient_front: "Server → Front",
      synchronization: "Federated sync",
    };
    return (
      <>
        <div className="chart-grid">
          <Chart
            title="Measured tensor traffic"
            subtitle="Actual transmitted tensor bytes, including synchronization"
            dark={dark}
            option={
              entries.length
                ? {
                    tooltip: {
                      trigger: "item",
                      valueFormatter: (v: any) => bytes(v),
                    },
                    legend: {
                      bottom: 0,
                      textStyle: { color: dark ? "#94a1b6" : "#657187" },
                    },
                    series: [
                      {
                        type: "pie",
                        radius: ["45%", "71%"],
                        center: ["50%", "44%"],
                        label: { show: false },
                        data: entries.map(([name, value]) => ({
                          name: stageNames[name] || title(name),
                          value,
                        })),
                      },
                    ],
                  }
                : null
            }
          />
          <Chart
            title="Communication by round"
            subtitle="Measured byte count · all U-shaped exchanges"
            option={lineOption(b, c, "communication_bytes", "Bytes")}
            dark={dark}
          />
        </div>
        <Chart
          title="U-shaped communication flow"
          subtitle="Arrow width represents measured bytes; aggregation traffic is shown separately"
          dark={dark}
          height={300}
          option={
            entries.length
              ? {
                  tooltip: { trigger: "item" },
                  series: [
                    {
                      type: "sankey",
                      left: 20,
                      right: 20,
                      top: 20,
                      bottom: 35,
                      emphasis: { focus: "adjacency" },
                      lineStyle: { color: "source", opacity: 0.35 },
                      label: { color: dark ? "#c7d2e2" : "#34445c" },
                      data: [
                        { name: "Client front" },
                        { name: "Server forward" },
                        { name: "Client back" },
                        { name: "Server backward" },
                        { name: "Front update" },
                        { name: "Federated server" },
                        { name: "Global redistribution" },
                      ],
                      links: [
                        {
                          source: "Client front",
                          target: "Server forward",
                          value: comm.activation || 0,
                        },
                        {
                          source: "Server forward",
                          target: "Client back",
                          value: comm.intermediate || 0,
                        },
                        {
                          source: "Client back",
                          target: "Server backward",
                          value: comm.gradient_back || 0,
                        },
                        {
                          source: "Server backward",
                          target: "Front update",
                          value: comm.gradient_front || 0,
                        },
                        {
                          source: "Federated server",
                          target: "Global redistribution",
                          value: comm.synchronization || 0,
                        },
                      ].filter((l) => l.value > 0),
                    },
                  ],
                }
              : null
          }
        />
        <div className="note">
          <Info size={16} />
          <span>
            Communication volume is measured from tensors. Network delay and
            energy are estimates derived from these measured bytes.
          </span>
        </div>
      </>
    );
  }
  if (tab === "Scalability")
    return (
      <SuitePanel
        suite={suites.scalability}
        name="Client and server scalability"
        dark={dark}
      />
    );
  if (tab === "Network") {
    const suite = suites.network;
    const rows: Data[] = suite?.rows || [];
    const heat: Data[] = (suite?.heatmap || []).filter(
      (r: Data) => r.variant === networkVariant,
    );
    const bandwidths = [...new Set(heat.map((r) => r.bandwidth_mbps))];
    const losses = [...new Set(heat.map((r) => r.packet_loss))];
    return (
      <>
        <div className="result-toolbar">
          <h3>Network stress testing</h3>
          <Badge status={suite?.status || "not_run"} />
        </div>
        <div className="chart-grid">
          <Chart
            title="Network latency"
            subtitle="Simulated transmission, retransmission and propagation · seconds"
            dark={dark}
            option={
              rows.length
                ? {
                    xAxis: {
                      type: "category",
                      data: rows.map(
                        (r) => `${r.scenario || r.label} ${r.variant || ""}`,
                      ),
                      axisLabel: { rotate: 25 },
                    },
                    yAxis: { type: "value", name: "Latency (s)" },
                    series: [
                      {
                        type: "bar",
                        name: "Latency estimate",
                        data: rows.map(
                          (r) =>
                            r.metrics?.latency_seconds ?? r.latency_seconds,
                        ),
                        itemStyle: { color: colors.blue },
                      },
                    ],
                  }
                : null
            }
          />
          <Chart
            title="Network energy"
            subtitle="Estimated from power settings and transfer durations · joules"
            dark={dark}
            option={
              rows.length
                ? {
                    xAxis: {
                      type: "category",
                      data: rows.map(
                        (r) => `${r.scenario || r.label} ${r.variant || ""}`,
                      ),
                      axisLabel: { rotate: 25 },
                    },
                    yAxis: { type: "value", name: "Energy (J)" },
                    series: [
                      {
                        type: "bar",
                        name: "Energy estimate",
                        data: rows.map((r) => r.metrics?.energy_joules),
                        itemStyle: { color: colors.amber },
                      },
                    ],
                  }
                : null
            }
          />
        </div>
        {heat.length > 0 && (
          <>
            <div className="result-toolbar">
              <h3>Bandwidth × packet loss</h3>
              <div>
                <select
                  aria-label="Network heatmap variant"
                  value={networkVariant}
                  onChange={(e) => setNetworkVariant(e.target.value)}
                >
                  <option value="baseline">Baseline</option>
                  <option value="candidate">Candidate</option>
                </select>
                <select
                  aria-label="Network heatmap metric"
                  value={networkMetric}
                  onChange={(e) => setNetworkMetric(e.target.value)}
                >
                  <option value="latency_seconds">Estimated latency (s)</option>
                  <option value="energy_joules">Estimated energy (J)</option>
                </select>
              </div>
            </div>
            <Chart
              title="Network cost heatmap"
              subtitle="Simulated cost replay of measured bytes; accuracy under packet drops is not measured"
              dark={dark}
              option={{
                grid: { left: 90, right: 30, top: 30, bottom: 90 },
                xAxis: {
                  type: "category",
                  name: "Packet loss",
                  data: losses.map((p) => `${100 * p}%`),
                },
                yAxis: {
                  type: "category",
                  name: "Bandwidth (Mbps)",
                  data: bandwidths,
                },
                visualMap: {
                  min: Math.min(...heat.map((r) => r[networkMetric])),
                  max: Math.max(...heat.map((r) => r[networkMetric])),
                  calculable: true,
                  orient: "horizontal",
                  left: "center",
                  bottom: 0,
                  inRange: { color: ["#273551", "#496f8d", "#67d9bf"] },
                },
                series: [
                  {
                    type: "heatmap",
                    data: heat.map((r) => [
                      losses.indexOf(r.packet_loss),
                      bandwidths.indexOf(r.bandwidth_mbps),
                      r[networkMetric],
                    ]),
                    label: {
                      show: true,
                      formatter: (p: any) => fmt(p.value[2], 1),
                      color: "#fff",
                    },
                  },
                ],
              }}
            />
          </>
        )}
        <section className="panel">
          <div className="section-heading">
            <h3>Transport assumptions</h3>
            <span className="data-kind estimate">SIMULATED</span>
          </div>
          <dl className="definition-list">
            {Object.entries(experiment?.config.network || {}).map(([k, v]) => (
              <div key={k}>
                <dt>{title(k)}</dt>
                <dd>{String(v)}</dd>
              </div>
            ))}
          </dl>
        </section>
        <Limitations
          items={
            suite?.limitations || [
              "Network stress estimates reuse measured tensor sizes. They do not imply measured distributed training performance.",
            ]
          }
        />
      </>
    );
  }
  if (tab === "Parameters") {
    const suite = suites.sweep || suites.parameter_sweep,
      rows: Data[] = suite?.rows || [];
    const primaryValues = [...new Set(rows.map((r) => r.value))];
    const secondaryValues = suite?.secondary_parameter
      ? [...new Set(rows.map((r) => r.secondary_value))]
      : ["Validation"];
    return (
      <>
        <SuitePanel
          suite={suite}
          name="Parameter sensitivity"
          dark={dark}
          parameter="value"
        />
        {rows.length > 0 && (
          <Chart
            title="Validation parameter map"
            subtitle="Color encodes observed validation accuracy (%) · test results do not select parameters"
            dark={dark}
            height={suite?.secondary_parameter ? 300 : 220}
            option={{
              tooltip: { position: "top" },
              grid: { top: 20, bottom: 70, left: 80, right: 25 },
              xAxis: {
                type: "category",
                name: suite?.parameter,
                data: primaryValues.map(String),
              },
              yAxis: {
                type: "category",
                name: suite?.secondary_parameter || "",
                data: secondaryValues.map(String),
              },
              visualMap: {
                min: 0,
                max: 100,
                calculable: true,
                orient: "horizontal",
                left: "center",
                bottom: 0,
                inRange: { color: ["#273551", "#496f8d", "#67d9bf"] },
                textStyle: { color: dark ? "#94a1b6" : "#657187" },
              },
              series: [
                {
                  type: "heatmap",
                  data: rows.map((r) => [
                    primaryValues.indexOf(r.value),
                    suite?.secondary_parameter
                      ? secondaryValues.indexOf(r.secondary_value)
                      : 0,
                    r.metrics?.val_accuracy ?? null,
                  ]),
                  label: {
                    show: true,
                    formatter: (p: any) =>
                      numeric(p.value[2]) ? `${fmt(p.value[2])}%` : "—",
                    color: "#fff",
                  },
                },
              ],
            }}
          />
        )}
      </>
    );
  }
  if (tab === "Ablation")
    return (
      <SuitePanel
        suite={suites.ablation}
        name="Component ablation"
        dark={dark}
      />
    );
  if (tab === "Statistics") {
    const distributions = [
      b.map((r) => r.metrics?.test_accuracy).filter(numeric),
      c.map((r) => r.metrics?.test_accuracy).filter(numeric),
    ];
    function quantile(values: number[], p: number) {
      const sorted = [...values].sort((a, b) => a - b),
        pos = (sorted.length - 1) * p,
        lo = Math.floor(pos);
      return sorted[lo] + (sorted[Math.ceil(pos)] - sorted[lo]) * (pos - lo);
    }
    const box = distributions.map((v) =>
      v.length
        ? [
            Math.min(...v),
            quantile(v, 0.25),
            quantile(v, 0.5),
            quantile(v, 0.75),
            Math.max(...v),
          ]
        : [],
    );
    return (
      <>
        {comparisonNotice}
        <Chart
          title="Repeated-seed test accuracy"
          subtitle="Whiskers show observed minimum and maximum; boxes show quartiles and median"
          dark={dark}
          option={
            distributions.some((v) => v.length)
              ? {
                  xAxis: { type: "category", data: ["Baseline", "Candidate"] },
                  yAxis: {
                    type: "value",
                    name: "Accuracy (%)",
                    min: 0,
                    max: 100,
                  },
                  series: [
                    {
                      type: "boxplot",
                      data: box.map((v, i) => ({
                        value: v,
                        itemStyle: {
                          color: i
                            ? colors.candidate + "33"
                            : colors.baseline + "33",
                          borderColor: i ? colors.candidate : colors.baseline,
                        },
                      })),
                    },
                    {
                      type: "scatter",
                      symbolSize: 7,
                      data: distributions.flatMap((v, i) =>
                        v.map((n) => [i, n]),
                      ),
                      itemStyle: { color: colors.amber },
                    },
                  ],
                }
              : null
          }
        />
        <div className="table-wrap panel">
          <table>
            <thead>
              <tr>
                <th>Metric</th>
                <th>Paired seeds</th>
                <th>Mean Δ</th>
                <th>95% CI of Δ</th>
                <th>Holm-adjusted p</th>
                <th>Interpretation</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(comp.statistics || {}).map(
                ([k, s]: [string, any]) => (
                  <tr key={k}>
                    <td>{metricNames[k] || title(k)}</td>
                    <td>{s.n_pairs ?? "—"}</td>
                    <td>{fmt(s.mean_difference, 4)}</td>
                    <td>
                      {s.ci95
                        ? `${fmt(s.ci95[0], 4)} to ${fmt(s.ci95[1], 4)}`
                        : "Unavailable"}
                    </td>
                    <td
                      title={`Raw p: ${fmt(s.p_value, 5)} · Paired effect dz: ${fmt(s.effect_size, 4)}`}
                    >
                      {fmt(s.adjusted_p_value, 5)}
                    </td>
                    <td>{title(s.status)}</td>
                  </tr>
                ),
              )}
            </tbody>
          </table>
        </div>
        <Limitations
          items={[
            "A small seed count gives weak statistical evidence. Confidence intervals and tests are interpreted with effect size, acceptance criteria, and experimental limitations.",
            "Exploratory comparisons across many metrics are not confirmatory evidence without a prespecified multiplicity strategy.",
          ]}
        />
      </>
    );
  }
  if (tab === "Attacks") {
    const attack = suites.attacks || {},
      rows: Data[] = attack.rows || [];
    return (
      <>
        <div className="result-toolbar">
          <div>
            <h3>Privacy attack probes</h3>
            <p>
              Bounded reconstruction optimization on a shared trained reference.
              Metrics summarize {attack.samples || 1} example(s); images show
              the first.
            </p>
          </div>
          <Badge status={attack.status || "not_run"} />
        </div>
        {rows.length ? (
          <>
            <div className="attack-grid">
              {attack.source_png && (
                <div className="attack-card panel">
                  <h4>Source example</h4>
                  <img
                    src={attack.source_png}
                    alt="Source training example used by attack probe"
                  />
                  <p>Training example · label {attack.source_label}</p>
                </div>
              )}
              {rows.map((r: Data) => (
                <div key={r.condition} className="attack-card panel">
                  <h4>{r.condition}</h4>
                  <div className="attack-images">
                    {["feature", "gradient"].map((kind) => (
                      <div key={kind}>
                        {r[kind]?.reconstruction_png ? (
                          <img
                            src={r[kind].reconstruction_png}
                            alt={`${r.condition} ${kind} reconstruction`}
                          />
                        ) : (
                          <span>Unavailable</span>
                        )}
                        <small>{title(kind)} attack</small>
                        <dl>
                          <dt>MSE</dt>
                          <dd>{fmt(r[kind]?.mse, 5)}</dd>
                          <dt>PSNR</dt>
                          <dd>{fmt(r[kind]?.psnr_db)} dB</dd>
                          <dt>SSIM</dt>
                          <dd>{fmt(r[kind]?.ssim, 4)}</dd>
                        </dl>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
            <p className="caption">
              Lower reconstruction MSE and higher PSNR / SSIM indicate a more
              successful attack under this probe. They are not privacy
              guarantees.
            </p>
          </>
        ) : (
          <Empty
            title="Attack probes not run"
            detail={
              attack.reason ||
              "Enable attacks and run the full evaluation to optimize real reconstructions."
            }
          />
        )}
        <Limitations
          items={[...(attack.assumptions || []), ...(attack.limitations || [])]}
        />
      </>
    );
  }
  const paretoInputs: Data[] = [
    ...(comp.pareto || []),
    ...(suites.sweep?.pareto || []),
  ].filter((p) => numeric(p[paretoMetric]) && numeric(p.val_accuracy));
  const pareto: Data[] = paretoInputs.map((p) => ({
    ...p,
    non_dominated: !paretoInputs.some(
      (q) =>
        q[paretoMetric] <= p[paretoMetric] &&
        q.val_accuracy >= p.val_accuracy &&
        (q[paretoMetric] < p[paretoMetric] || q.val_accuracy > p.val_accuracy),
    ),
  }));
  return (
    <>
      <section className="report-header panel">
        <span className="eyebrow">FORMULA IMPACT REPORT</span>
        <h2>{experiment?.name}</h2>
        <p>{experiment?.config.hypothesis}</p>
        <div className="report-badges">
          <Badge status={comp.hypothesis_status || "inconclusive"} />
          <span className="badge muted">{experiment?.id}</span>
          <span className="badge muted">
            {experiment?.config.seeds.length} paired seeds requested
          </span>
        </div>
        <div className="export-buttons">
          {["pdf", "csv", "json", "markdown"].map((f) => (
            <a
              className="button secondary"
              href={`/api/experiments/${experiment?.id}/export?format=${f}`}
              key={f}
              download
            >
              <FileDown size={15} />
              {f === "markdown" ? "Markdown" : f.toUpperCase()}
            </a>
          ))}
          <button className="button primary" onClick={onEvidence}>
            <Eye size={15} /> Open evidence
          </button>
        </div>
      </section>
      <section className="panel">
        <div className="section-heading">
          <h3>Measured impact</h3>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Metric</th>
                <th>Baseline mean</th>
                <th>Candidate mean</th>
                <th>Absolute change</th>
                <th>Relative change</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(comp.deltas || {}).map(
                ([key, d]: [string, any]) => (
                  <tr key={key}>
                    <td>{metricNames[key] || title(key)}</td>
                    <td>{metric(aggregate("baseline", key), key)}</td>
                    <td>{metric(aggregate("candidate", key), key)}</td>
                    <td>
                      {numeric(d.absolute) && d.absolute > 0 ? "+" : ""}
                      {fmt(d.absolute, 4)} {d.unit}
                    </td>
                    <td>
                      {numeric(d.relative_pct)
                        ? `${d.relative_pct > 0 ? "+" : ""}${fmt(d.relative_pct)}%`
                        : "Undefined"}
                    </td>
                  </tr>
                ),
              )}
            </tbody>
          </table>
        </div>
      </section>
      <div className="result-toolbar">
        <h3>Pareto trade-offs</h3>
        <select
          aria-label="Pareto horizontal objective"
          value={paretoMetric}
          onChange={(e) => setParetoMetric(e.target.value)}
        >
          {["communication_bytes", "energy_joules", "epsilon"].map((k) => (
            <option key={k} value={k}>
              {metricNames[k]} vs validation accuracy
            </option>
          ))}
        </select>
      </div>
      <Chart
        title="Observed objective frontier"
        subtitle="Lower horizontal cost and higher validation accuracy are preferred · only evaluated configurations"
        dark={dark}
        option={
          pareto.some(
            (p) => numeric(p[paretoMetric]) && numeric(p.val_accuracy),
          )
            ? {
                tooltip: {
                  trigger: "item",
                  formatter: (p: any) =>
                    `${p.data.name}<br/>${metricNames[paretoMetric]}: ${metric(p.value[0], paretoMetric)}<br/>Validation: ${fmt(p.value[1])}%`,
                },
                xAxis: { type: "value", name: metricUnits[paretoMetric] },
                yAxis: {
                  type: "value",
                  name: "Validation (%)",
                  min: 0,
                  max: 100,
                },
                series: [
                  {
                    type: "scatter",
                    symbolSize: 15,
                    data: pareto
                      .filter(
                        (p) =>
                          numeric(p[paretoMetric]) && numeric(p.val_accuracy),
                      )
                      .map((p) => ({
                        name: p.label,
                        value: [p[paretoMetric], p.val_accuracy],
                        itemStyle: {
                          color: p.non_dominated
                            ? colors.candidate
                            : colors.baseline,
                        },
                        label: {
                          show: true,
                          position: "top",
                          formatter: p.label,
                          color: dark ? "#c7d2e2" : "#34445c",
                        },
                      })),
                  },
                ],
              }
            : null
        }
      />
      <Limitations
        items={[...(result.limitations || []), ...(comp.limitations || [])]}
      />
    </>
  );
}
function ChartIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
      <path
        d="M2 11V3M2 11H12M4 8L7 5L9 7L12 2"
        stroke="currentColor"
        strokeWidth="1.4"
      />
    </svg>
  );
}
function RunTable({
  baseline,
  candidate,
}: {
  baseline: Data[];
  candidate: Data[];
}) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Variant / seed</th>
            <th>Train</th>
            <th>Validation</th>
            <th>Test</th>
            <th>Test loss</th>
            <th>Runtime</th>
          </tr>
        </thead>
        <tbody>
          {[
            ["Baseline", baseline],
            ["Candidate", candidate],
          ].flatMap(([variant, runs]) =>
            (runs as Data[]).map((r, i) => (
              <tr key={`${variant}-${i}`}>
                <td>
                  <span
                    className={`legend-dot ${variant === "Candidate" ? "candidate" : ""}`}
                  />
                  {String(variant)} · {r.seed ?? i + 1}
                </td>
                <td>{metric(r.metrics?.train_accuracy, "train_accuracy")}</td>
                <td>{metric(r.metrics?.val_accuracy, "val_accuracy")}</td>
                <td>{metric(r.metrics?.test_accuracy, "test_accuracy")}</td>
                <td>{fmt(r.metrics?.test_loss, 4)}</td>
                <td>{metric(r.metrics?.runtime_seconds, "runtime_seconds")}</td>
              </tr>
            )),
          )}
        </tbody>
      </table>
    </div>
  );
}
