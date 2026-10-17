import { useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  ArrowUpRight,
  Beaker,
  BookOpen,
  Check,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Clock3,
  Code2,
  Copy,
  Database,
  FileText,
  FlaskConical,
  History,
  Layers3,
  LoaderCircle,
  Menu,
  Moon,
  Network,
  Play,
  Plus,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Square,
  Sun,
  Terminal,
  Trash2,
  WandSparkles,
  X,
} from "lucide-react";
import Results, { Badge, Checks, SplitDiagram } from "./Results";
import { api, download, fmt, jsonOptions, numeric, tabs, title } from "./types";
import type { Data, Experiment } from "./types";

type FieldProps = {
  label: string;
  path: string;
  config: Data;
  set: (path: string, value: any) => void;
  type?: string;
  options?: [string, string][];
  min?: number;
  max?: number;
  step?: number | string;
  tip?: string;
  placeholder?: string;
};
function Field({
  label,
  path,
  config,
  set,
  type = "number",
  options,
  min,
  max,
  step,
  tip,
  placeholder,
}: FieldProps) {
  const value = path.split(".").reduce<any>((obj, key) => obj?.[key], config);
  return (
    <label className="field">
      <span title={tip}>
        {label}
        {tip && <CircleHelp size={11} />}
      </span>
      {options ? (
        <select
          aria-label={label}
          value={value ?? ""}
          onChange={(e) => set(path, e.target.value)}
        >
          {options.map(([v, t]) => (
            <option value={v} key={v}>
              {t}
            </option>
          ))}
        </select>
      ) : (
        <input
          aria-label={label}
          type={type}
          value={value ?? ""}
          min={min}
          max={max}
          step={step ?? "any"}
          placeholder={placeholder}
          onChange={(e) =>
            set(
              path,
              type === "number"
                ? e.target.value === ""
                  ? null
                  : Number(e.target.value)
                : e.target.value,
            )
          }
        />
      )}
    </label>
  );
}
function Toggle({
  label,
  description,
  checked,
  onChange,
}: {
  label: string;
  description?: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="toggle-row">
      <span>
        <strong>{label}</strong>
        {description && <small>{description}</small>}
      </span>
      <input
        aria-label={label}
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="toggle" />
    </label>
  );
}
function NumberList({
  label,
  value,
  onChange,
  min = 0,
}: {
  label: string;
  value: number[];
  onChange: (v: number[]) => void;
  min?: number;
}) {
  const [draft, setDraft] = useState(value.join(", ")),
    [error, setError] = useState(false);
  useEffect(() => setDraft(value.join(", ")), [JSON.stringify(value)]);
  return (
    <label className="field">
      <span>{label}</span>
      <input
        aria-label={label}
        value={draft}
        onChange={(e) => {
          setDraft(e.target.value);
          const v = e.target.value.split(",").map((s) => Number(s.trim()));
          const good =
            e.target.value.trim().length > 0 &&
            v.every((n) => numeric(n) && n >= min);
          setError(!good);
          if (good) onChange(v);
        }}
        aria-invalid={error}
      />
      {error && (
        <small className="error-text">
          Enter comma-separated numbers ≥ {min}.
        </small>
      )}
    </label>
  );
}
const stages = [
  "Dataset preparation",
  "Baseline training",
  "Candidate training",
  "Comparison",
  "Stress testing",
  "Final report",
];
function phaseIndex(phase: string) {
  if (/report|complete|final/i.test(phase)) return 5;
  if (
    /suite|attack|network|ablation|sweep|non.?iid|scalab|split.profile/i.test(
      phase,
    )
  )
    return 4;
  if (/compar|statistic/i.test(phase)) return 3;
  if (/candidate/i.test(phase)) return 2;
  if (/baseline/i.test(phase)) return 1;
  return 0;
}
export default function App() {
  const [dark, setDark] = useState(
    () => localStorage.getItem("sfl-theme") !== "light",
  );
  const [catalog, setCatalog] = useState<Data | null>(null),
    [config, setConfig] = useState<Data | null>(null),
    [experiments, setExperiments] = useState<Experiment[]>([]),
    [selected, setSelected] = useState<Experiment>(),
    [tab, setTab] = useState("Overview"),
    [connection, setConnection] = useState("connecting"),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(""),
    [validation, setValidation] = useState<Data | null>(null),
    [keyboard, setKeyboard] = useState("Basic"),
    [historyOpen, setHistoryOpen] = useState(false),
    [evidenceOpen, setEvidenceOpen] = useState(false),
    [helpOpen, setHelpOpen] = useState(false),
    [evidenceTab, setEvidenceTab] = useState("Configuration"),
    [menuOpen, setMenuOpen] = useState(false),
    [parameterName, setParameterName] = useState("");
  const editor = useRef<HTMLTextAreaElement>(null),
    resultsRef = useRef<HTMLElement>(null),
    setupRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    localStorage.setItem("sfl-theme", dark ? "dark" : "light");
  }, [dark]);
  async function initialize() {
    setConnection("connecting");
    setError("");
    const settled = await Promise.allSettled([
      api("/health"),
      api("/catalog"),
      api("/experiments"),
    ]);
    if (settled[0].status === "fulfilled") setConnection("online");
    else {
      setConnection("offline");
      setError(
        "The research API is unavailable. Start the backend, then reconnect.",
      );
    }
    if (settled[1].status === "fulfilled") {
      const data = settled[1].value;
      setCatalog(data);
      setConfig((c) => c || structuredClone(data.defaults));
    } else setError(String(settled[1].reason));
    if (settled[2].status === "fulfilled") {
      const list = Array.isArray(settled[2].value)
        ? settled[2].value
        : settled[2].value.experiments || [];
      setExperiments(list);
    }
  }
  useEffect(() => {
    void initialize();
  }, []);
  useEffect(() => {
    if (!selected || !["queued", "running"].includes(selected.status)) return;
    let disposed = false;
    const timer = setInterval(async () => {
      try {
        const record = await api(`/experiments/${selected.id}`);
        if (disposed) return;
        setSelected(record);
        setExperiments((list) => [
          record,
          ...list.filter((e) => e.id !== record.id),
        ]);
        setConnection("online");
      } catch (e) {
        if (!disposed) {
          setConnection("offline");
          setError(`Progress refresh failed: ${String(e)}`);
        }
      }
    }, 1500);
    return () => {
      disposed = true;
      clearInterval(timer);
    };
  }, [selected?.id, selected?.status]);
  const active = !!selected && ["queued", "running"].includes(selected.status);
  function set(path: string, value: any) {
    setConfig((current) => {
      if (!current) return current;
      const next = structuredClone(current);
      const bits = path.split(".");
      let target = next;
      for (const key of bits.slice(0, -1)) target = target[key];
      target[bits[bits.length - 1]] = value;
      if (path === "mode" && value === "paper_reproduction") {
        next.dataset = "imagefolder";
        next.model = "resnet18";
        next.split = [2, 4, 2];
        next.rounds = 100;
        next.local_epochs = 1;
      }
      return next;
    });
    if (path === "candidate" || path.startsWith("candidate."))
      setValidation(null);
  }
  const candidate = config?.candidate || {},
    hook = catalog?.hooks?.find((h: Data) => h.id === candidate.hook);
  const fields = (
    path: string,
    label: string,
    extra: Partial<FieldProps> = {},
  ) => (
    <Field
      key={path}
      label={label}
      path={path}
      config={config!}
      set={set}
      {...extra}
    />
  );
  function insert(value: string) {
    const input = editor.current;
    const expression = candidate.expression || "";
    const start = input?.selectionStart ?? expression.length,
      end = input?.selectionEnd ?? start;
    set(
      "candidate.expression",
      expression.slice(0, start) + value + expression.slice(end),
    );
    requestAnimationFrame(() => {
      input?.focus();
      input?.setSelectionRange(start + value.length, start + value.length);
    });
  }
  function template(id: string) {
    const t = catalog?.templates?.find((item: Data) => item.id === id);
    if (!t) return;
    set("candidate", {
      ...candidate,
      name: t.name,
      hook: t.hook,
      expression: t.expression,
      parameters: structuredClone(t.parameters),
      mode: "formula",
    });
  }
  function setMode(mode: string) {
    set("candidate", {
      ...candidate,
      mode,
      algorithm:
        mode === "algorithm" && !candidate.algorithm.length
          ? structuredClone(catalog?.default_algorithm || [])
          : candidate.algorithm,
    });
  }
  async function validate(): Promise<Data | null> {
    setError("");
    setBusy("validate");
    try {
      const result = await api("/validate", jsonOptions(candidate));
      setValidation(result);
      return result;
    } catch (e) {
      setError(String(e));
      return null;
    } finally {
      setBusy("");
    }
  }
  async function run(action: string) {
    if (!config) return;
    setError("");
    if (config.hypothesis.trim().length < 10) {
      setError(
        "Define a research hypothesis of at least 10 characters before running.",
      );
      return;
    }
    if (action !== "baseline") {
      const valid = await validate();
      if (!valid?.valid) return;
    }
    setBusy(action);
    try {
      const record = await api("/experiments", jsonOptions({ config, action }));
      setSelected(record);
      setExperiments((list) => [
        record,
        ...list.filter((e) => e.id !== record.id),
      ]);
      setTab("Overview");
      setHistoryOpen(false);
      setTimeout(
        () =>
          resultsRef.current?.scrollIntoView({
            behavior: "smooth",
            block: "start",
          }),
        100,
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy("");
    }
  }
  async function cancel() {
    if (!selected) return;
    setBusy("cancel");
    try {
      await api(`/experiments/${selected.id}/cancel`, {
        method: "POST",
      });
      setSelected(await api(`/experiments/${selected.id}`));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy("");
    }
  }
  async function selectExperiment(e: Experiment) {
    try {
      setSelected(await api(`/experiments/${e.id}`));
      setTab("Overview");
      setHistoryOpen(false);
      setTimeout(
        () => resultsRef.current?.scrollIntoView({ behavior: "smooth" }),
        50,
      );
    } catch (err) {
      setError(String(err));
    }
  }
  function addOperation(op: string) {
    const definition = catalog?.operations?.find((o: Data) => o.op === op);
    if (!definition) return;
    const blocks = [
      ...candidate.algorithm,
      { op, params: structuredClone(definition.params || {}) },
    ];
    blocks.sort(
      (a, b) =>
        (catalog?.operations?.find((o: Data) => o.op === a.op)?.stage ?? 0) -
        (catalog?.operations?.find((o: Data) => o.op === b.op)?.stage ?? 0),
    );
    set("candidate.algorithm", blocks);
  }
  function moveOperation(index: number, delta: number) {
    const blocks = [...candidate.algorithm];
    const target = index + delta;
    if (target < 0 || target >= blocks.length) return;
    const stage = (op: string) =>
      catalog?.operations?.find((o: Data) => o.op === op)?.stage;
    if (stage(blocks[index].op) !== stage(blocks[target].op)) return;
    [blocks[index], blocks[target]] = [blocks[target], blocks[index]];
    set("candidate.algorithm", blocks);
  }
  function newExperiment() {
    if (catalog?.defaults) setConfig(structuredClone(catalog.defaults));
    setValidation(null);
    setSelected(undefined);
    setError("");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
  const evidence = selected
    ? evidenceTab === "Configuration"
      ? selected.config
      : evidenceTab === "Results"
        ? selected.result
        : evidenceTab === "Logs"
          ? selected.logs
          : {
              id: selected.id,
              created_at: selected.created_at,
              status: selected.status,
              error: selected.error,
              environment: selected.result?.environment,
            }
    : {};
  return (
    <div className="app-shell">
      <aside className={`sidebar ${menuOpen ? "mobile-open" : ""}`}>
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            window.scrollTo({ top: 0, behavior: "smooth" });
          }}
        >
          <span className="brand-mark">
            <Layers3 size={25} />
          </span>
          <span>
            SFL<span className="brand-light">-Lab</span>
            <small>RESEARCH WORKBENCH</small>
          </span>
        </a>
        <button
          className="workspace-selector"
          onClick={() => setHelpOpen(true)}
        >
          <span className="workspace-avatar">R</span>
          <span>
            Research workspace<small>Local simulation environment</small>
          </span>
          <ChevronDown size={13} />
        </button>
        <div className="nav-caption">WORKSPACE</div>
        <nav>
          <button
            className={!historyOpen ? "nav-item selected" : "nav-item"}
            onClick={() => {
              setHistoryOpen(false);
              setMenuOpen(false);
              window.scrollTo({ top: 0, behavior: "smooth" });
            }}
          >
            <FlaskConical size={17} /> Experiment workbench
            <span className="nav-dot" />
          </button>
          <button
            className="nav-item"
            onClick={() => {
              setMenuOpen(false);
              resultsRef.current?.scrollIntoView({ behavior: "smooth" });
            }}
          >
            <Activity size={17} /> Results & analysis
          </button>
          <button
            className={historyOpen ? "nav-item selected" : "nav-item"}
            onClick={() => {
              setHistoryOpen(true);
              setMenuOpen(false);
            }}
          >
            <History size={17} /> Experiment history
            <span className="nav-count">{experiments.length}</span>
          </button>
        </nav>
        <div className="nav-caption">REFERENCE</div>
        <nav>
          <button
            className="nav-item"
            onClick={() => {
              setHelpOpen(true);
              setMenuOpen(false);
            }}
          >
            <BookOpen size={17} /> Scientific methodology
          </button>
          <button
            className="nav-item"
            onClick={() => {
              setTab("Architecture");
              resultsRef.current?.scrollIntoView({ behavior: "smooth" });
              setMenuOpen(false);
            }}
          >
            <Network size={17} /> SFL architecture
          </button>
        </nav>
        <div className="sidebar-bottom">
          <div className="integrity-note">
            <ShieldCheck size={20} />
            <strong>Evidence, by design.</strong>
            <p>
              Controlled baselines.
              <br />
              Explicit assumptions.
              <br />
              Measured conclusions.
            </p>
          </div>
          <div className="sidebar-version">
            <span className="status-dot" /> SFL-Lab <span>v0.1</span>
          </div>
        </div>
      </aside>
      {menuOpen && (
        <div className="mobile-scrim" onClick={() => setMenuOpen(false)} />
      )}
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            <button
              className="icon-button mobile-menu"
              aria-label="Open navigation"
              onClick={() => setMenuOpen(true)}
            >
              <Menu size={19} />
            </button>
            <span>Workspace</span>
            <ChevronRight size={12} />
            <strong>Experiment workbench</strong>
          </div>
          <div className="topbar-actions">
            <button
              className={`connection ${connection}`}
              onClick={() => void initialize()}
              title="Reconnect to the research API"
            >
              <i />
              {connection === "online"
                ? "Engine connected"
                : connection === "connecting"
                  ? "Connecting…"
                  : "Reconnect engine"}
            </button>
            <span className="topbar-divider" />
            <button
              className="icon-button"
              aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
              onClick={() => setDark(!dark)}
            >
              {dark ? <Sun size={17} /> : <Moon size={17} />}
            </button>
            <button
              className="icon-button"
              aria-label="Open help"
              onClick={() => setHelpOpen(true)}
            >
              <CircleHelp size={17} />
            </button>
            <span className="user-avatar">R</span>
          </div>
        </header>
        <main>
          <section className="page-heading">
            <div>
              <div className="eyebrow">
                <span className="tiny-green-line" /> SPLIT FEDERATED LEARNING
                LABORATORY
              </div>
              <h1>
                From hypothesis to evidence<span>.</span>
              </h1>
              <p>
                Explore a formula. Preserve the architecture. Measure what
                changes.
              </p>
            </div>
            <button
              className="button secondary"
              onClick={newExperiment}
              disabled={active}
            >
              <Plus size={15} /> New experiment
            </button>
          </section>
          <div className="workflow-strip">
            <span>
              <Code2 size={14} /> Formula / algorithm
            </span>
            <ArrowRight size={14} />
            <span>
              <ShieldCheck size={14} /> Validate
            </span>
            <ArrowRight size={14} />
            <span>
              <Layers3 size={14} /> Integrate
            </span>
            <ArrowRight size={14} />
            <span>
              <FlaskConical size={14} /> Compare
            </span>
            <ArrowRight size={14} />
            <span>
              <Activity size={14} /> Scientific conclusion
            </span>
            <span className="workflow-mode">CONTROLLED EXPERIMENTS</span>
          </div>
          {error && (
            <div className="error-banner" role="alert">
              <CircleHelp size={18} />
              <div>
                <strong>Action needs attention</strong>
                <p>{error}</p>
              </div>
              <button
                className="icon-button"
                aria-label="Dismiss error"
                onClick={() => setError("")}
              >
                <X size={16} />
              </button>
            </div>
          )}
          {!config ? (
            <div className="loading-panel panel">
              <LoaderCircle className="spin" size={25} />
              <h3>
                {connection === "offline"
                  ? "Waiting for the research API"
                  : "Loading your workbench"}
              </h3>
              <p>
                The backend supplies experiment defaults and the validated
                operation catalog.
              </p>
              <button
                className="button secondary"
                onClick={() => void initialize()}
              >
                Reconnect
              </button>
            </div>
          ) : (
            <div className="setup-grid" ref={setupRef}>
              <section className="setup-panel panel">
                <div className="section-heading">
                  <div className="heading-with-icon">
                    <Settings2 size={16} />
                    <h2>Experiment setup</h2>
                  </div>
                  <span className="step-badge">01</span>
                </div>
                <div className="setup-content">
                  {fields("name", "Experiment name", { type: "text" })}
                  <div className="field-grid">
                    {fields("mode", "Research workflow", {
                      options: [
                        ["research", "Research mode"],
                        ["paper_reproduction", "Paper preset · local SeaShips"],
                      ],
                    })}
                    {fields("dataset", "Dataset", {
                      options: [
                        ["digits", "Digits · bundled"],
                        ["cifar10", "CIFAR-10 · download"],
                        ["imagefolder", "Local image folder"],
                      ],
                    })}
                  </div>
                  {config.mode === "paper_reproduction" && (
                    <div className="inline-note">
                      Paper preset requires SeaShips, ResNet18, a 2–4–2 split,
                      and independently verified original settings. It does not
                      establish numerical reproduction.
                    </div>
                  )}
                  {config.dataset === "imagefolder" &&
                    fields("dataset_path", "Local dataset path", {
                      type: "text",
                      placeholder: "/workspace/data/seaships",
                    })}
                  <div className="field-grid">
                    {fields("model", "Model architecture", {
                      options: [
                        ["resnet18", "ResNet18"],
                        ["tiny_cnn", "Tiny CNN · quick checks"],
                      ],
                    })}
                    {fields("samples", "Sample limit", {
                      min: 40,
                      max: 1000000,
                      step: 1,
                    })}
                  </div>
                  <div className="split-control">
                    <label>
                      Model split{" "}
                      <span title="Eight sequential network blocks are divided across client front, server middle, and client back.">
                        <CircleHelp size={11} />
                      </span>
                    </label>
                    <div>
                      {config.split.map((n: number, i: number) => (
                        <label key={i}>
                          <input
                            aria-label={
                              [
                                "Client front blocks",
                                "Server blocks",
                                "Client back blocks",
                              ][i]
                            }
                            type="number"
                            min={1}
                            max={6}
                            value={n}
                            onChange={(e) => {
                              const next = [...config.split];
                              next[i] = Number(e.target.value);
                              set("split", next);
                            }}
                          />
                          <span>
                            {["Client front", "Server", "Client back"][i]}
                          </span>
                          {i < 2 && <b>:</b>}
                        </label>
                      ))}
                    </div>
                    {config.split.reduce((a: number, b: number) => a + b, 0) !==
                      8 && (
                      <small className="error-text">
                        Split block counts must sum to 8.
                      </small>
                    )}
                  </div>
                  <div className="field-grid">
                    {fields("clients", "Clients", {
                      min: 2,
                      max: 100,
                      step: 1,
                      tip: "Independent logical data partitions in a single-process federated simulation.",
                    })}
                    {fields("servers", "Server replicas", {
                      min: 1,
                      max: 16,
                      step: 1,
                      tip: "Simulated server assignment and synchronization; no physical distributed speedup.",
                    })}
                    {fields("rounds", "Federated rounds", {
                      min: 1,
                      max: 1000,
                      step: 1,
                    })}
                    {fields("local_epochs", "Local epochs", {
                      min: 1,
                      max: 100,
                      step: 1,
                    })}
                    {fields("batch_size", "Batch size", {
                      min: 2,
                      max: 512,
                      step: 1,
                    })}
                    {fields("learning_rate", "Learning rate η", {
                      min: 0.000001,
                      max: 10,
                      step: 0.001,
                    })}
                    {fields("optimizer", "Optimizer", {
                      options: [
                        ["sgd", "SGD"],
                        ["adam", "Adam"],
                      ],
                    })}
                    {fields("partition", "Client partition", {
                      options: [
                        ["iid", "IID · balanced"],
                        ["dirichlet", "Non-IID · Dirichlet"],
                        ["label_skew", "Non-IID · label skew"],
                        ["quantity", "Quantity imbalance"],
                      ],
                    })}
                  </div>
                  {config.partition === "dirichlet" &&
                    fields("dirichlet_alpha", "Dirichlet α", {
                      min: 0.001,
                      tip: "Smaller alpha produces stronger label heterogeneity.",
                    })}
                  <NumberList
                    label="Random seeds · paired across variants"
                    value={config.seeds}
                    onChange={(v) => set("seeds", v)}
                  />
                  <div className="dataset-split">
                    <div>
                      <span style={{ width: "70%" }} />
                      <span style={{ width: "10%" }} />
                      <span style={{ width: "20%" }} />
                    </div>
                    <p>
                      <span>
                        <i />
                        Train 70%
                      </span>
                      <span>
                        <i />
                        Validation 10%
                      </span>
                      <span>
                        <i />
                        Test 20%
                      </span>
                    </p>
                  </div>
                  <details className="advanced">
                    <summary>
                      <ShieldCheck size={14} /> Privacy & candidate protections{" "}
                      <ChevronDown size={14} />
                    </summary>
                    <div className="advanced-body">
                      <div className="field-grid">
                        {fields(
                          "candidate.activation_clip",
                          "Activation clip Cₐ",
                          {
                            min: 0.000001,
                            tip: "Blank disables the additional activation clipping module.",
                          },
                        )}
                        {fields(
                          "candidate.activation_noise",
                          "Activation noise σₐ",
                          { min: 0 },
                        )}
                        {fields("candidate.gradient_clip", "Gradient clip Cɢ", {
                          min: 0.000001,
                          tip: "Blank disables the additional gradient clipping module.",
                        })}
                        {fields(
                          "candidate.gradient_noise",
                          "Gradient noise σɢ",
                          { min: 0 },
                        )}
                        {fields("candidate.fedprox_mu", "FedProx μ", {
                          min: 0,
                        })}
                        {fields("candidate.aggregation", "Aggregation", {
                          options: [
                            ["weighted", "Sample-weighted"],
                            ["equal", "Equal clients"],
                          ],
                        })}
                        {fields("privacy.delta", "Privacy δ", {
                          min: 0,
                          max: 1,
                        })}
                        {fields("privacy.epsilon_max", "Budget ε max", {
                          min: 0.001,
                        })}
                        {fields(
                          "privacy.warning_fraction",
                          "Warning fraction",
                          { min: 0.01, max: 0.99 },
                        )}
                      </div>
                      {fields("privacy.threat_model", "Threat model", {
                        options: [
                          [
                            "honest_but_curious_server",
                            "Honest-but-curious server",
                          ],
                          ["external_eavesdropper", "External eavesdropper"],
                          ["malicious_server", "Malicious server"],
                          ["malicious_client", "Malicious client"],
                          ["parameter_server", "Parameter server"],
                        ],
                      })}
                      <Toggle
                        label="Stop at privacy budget"
                        checked={config.privacy.stop_on_budget}
                        onChange={(v) => set("privacy.stop_on_budget", v)}
                      />
                      <p className="caption">
                        Accounted tensor transcript privacy is scoped; it is not
                        full-system differential privacy.
                      </p>
                    </div>
                  </details>
                  <details className="advanced">
                    <summary>
                      <Network size={14} /> Network & energy models{" "}
                      <ChevronDown size={14} />
                    </summary>
                    <div className="advanced-body">
                      <div className="field-grid">
                        {fields("network.bandwidth_mbps", "Bandwidth (Mbps)", {
                          min: 0.001,
                        })}
                        {fields("network.latency_ms", "Latency (ms)", {
                          min: 0,
                        })}
                        {fields("network.packet_loss", "Packet loss (0–1)", {
                          min: 0,
                          max: 0.99,
                        })}
                        {fields("network.jitter_ms", "Jitter (ms)", { min: 0 })}
                      </div>
                      <Toggle
                        label="Retransmission"
                        checked={config.network.retransmission}
                        onChange={(v) => set("network.retransmission", v)}
                      />
                      <Toggle
                        label="Energy estimates"
                        checked={config.energy.enabled}
                        onChange={(v) => set("energy.enabled", v)}
                      />
                      <div className="field-grid">
                        {fields("energy.frequency_ghz", "Frequency (GHz)", {
                          min: 0.01,
                        })}
                        {fields("energy.kappa", "Capacitance κ", { min: 0 })}
                        {fields(
                          "energy.cycles_per_second",
                          "Cycles per second",
                          { min: 1 },
                        )}
                        {fields("energy.tx_power_w", "Transmit power (W)", {
                          min: 0,
                        })}
                        {fields("energy.rx_power_w", "Receive power (W)", {
                          min: 0,
                        })}
                      </div>
                      <span className="data-kind estimate">
                        SIMULATED ESTIMATES
                      </span>
                    </div>
                  </details>
                  <details className="advanced">
                    <summary>
                      <SlidersHorizontal size={14} /> Acceptance criteria{" "}
                      <ChevronDown size={14} />
                    </summary>
                    <div className="advanced-body">
                      {fields(
                        "acceptance.max_accuracy_drop_pp",
                        "Maximum accuracy drop (percentage points)",
                        { min: 0 },
                      )}
                      {fields(
                        "acceptance.max_communication_increase_pct",
                        "Maximum communication increase (%)",
                        { min: 0 },
                      )}
                      {fields(
                        "acceptance.max_runtime_increase_pct",
                        "Maximum runtime increase (%)",
                        { min: 0 },
                      )}
                      {fields(
                        "acceptance.required_privacy_improvement_pct",
                        "Required privacy improvement (%)",
                        { min: 0, max: 100 },
                      )}
                      {fields("target_loss", "Target convergence loss", {
                        min: 0.00001,
                      })}
                      {fields("participation", "Client participation (0–1)", {
                        min: 0.01,
                        max: 1,
                      })}
                    </div>
                  </details>
                  <details className="advanced">
                    <summary>
                      <Beaker size={14} /> Full evaluation suites{" "}
                      <ChevronDown size={14} />
                    </summary>
                    <div className="advanced-body">
                      {[
                        ["network", "Network stress tests"],
                        ["non_iid", "Non-IID robustness"],
                        ["attacks", "Reconstruction attacks"],
                        ["ablation", "Component ablation"],
                        ["sweep", "Parameter sweep"],
                        ["scalability", "Client / server scaling"],
                        ["split_profile", "Split-point profiling"],
                      ].map(([key, label]) => (
                        <Toggle
                          key={key}
                          label={label}
                          checked={config.evaluation[key]}
                          onChange={(v) => set(`evaluation.${key}`, v)}
                        />
                      ))}
                      {fields(
                        "evaluation.attack_steps",
                        "Attack optimization steps",
                        { min: 1, max: 500, step: 1 },
                      )}
                      {fields("evaluation.sweep_parameter", "Sweep parameter", {
                        options: [
                          ["C_G", "Gradient clip Cɢ"],
                          ["sigma_G", "Gradient noise σɢ"],
                          ["mu", "FedProx μ"],
                        ],
                      })}
                      <NumberList
                        label="Sweep values"
                        value={config.evaluation.sweep_values}
                        onChange={(v) => set("evaluation.sweep_values", v)}
                      />
                      <label className="field">
                        <span>Second sweep dimension</span>
                        <select
                          value={
                            config.evaluation.sweep_secondary_parameter || ""
                          }
                          onChange={(e) =>
                            set(
                              "evaluation.sweep_secondary_parameter",
                              e.target.value || null,
                            )
                          }
                        >
                          <option value="">Disabled · one dimension</option>
                          {[
                            ["C_G", "Gradient clip Cɢ"],
                            ["sigma_G", "Gradient noise σɢ"],
                            ["mu", "FedProx μ"],
                          ]
                            .filter(
                              ([v]) => v !== config.evaluation.sweep_parameter,
                            )
                            .map(([v, label]) => (
                              <option key={v} value={v}>
                                {label}
                              </option>
                            ))}
                        </select>
                      </label>
                      {config.evaluation.sweep_secondary_parameter && (
                        <NumberList
                          label="Second dimension values"
                          value={
                            config.evaluation.sweep_secondary_values || [
                              0, 0.5, 1,
                            ]
                          }
                          onChange={(v) =>
                            set("evaluation.sweep_secondary_values", v)
                          }
                        />
                      )}
                      {fields("evaluation.search", "Search strategy", {
                        options: [
                          ["grid", "Grid search"],
                          ["random", "Seeded random search"],
                        ],
                      })}
                      <NumberList
                        label="Scalability client counts"
                        value={config.evaluation.client_counts}
                        onChange={(v) => set("evaluation.client_counts", v)}
                        min={2}
                      />
                      <NumberList
                        label="Scalability server counts"
                        value={config.evaluation.server_counts}
                        onChange={(v) => set("evaluation.server_counts", v)}
                        min={1}
                      />
                      <p className="caption">
                        Full evaluation executes additional conditions. Runtime
                        grows with rounds, seeds, and enabled suites.
                      </p>
                    </div>
                  </details>
                </div>
                <div className="setup-footer">
                  <ShieldCheck size={14} />
                  <span>
                    Identical splits, initialization and seeds
                    <br />
                    for baseline and candidate.
                  </span>
                </div>
              </section>
              <div className="editor-column">
                <section className="formula-panel panel">
                  <div className="section-heading">
                    <div className="heading-with-icon">
                      <Code2 size={17} />
                      <h2>Candidate workspace</h2>
                    </div>
                    <span className="step-badge">02</span>
                  </div>
                  <div className="editor-content">
                    <div className="mode-row">
                      <div className="segmented">
                        <button
                          className={
                            candidate.mode === "formula" ? "active" : ""
                          }
                          onClick={() => setMode("formula")}
                        >
                          <Code2 size={13} /> Formula
                        </button>
                        <button
                          className={
                            candidate.mode === "algorithm" ? "active" : ""
                          }
                          onClick={() => setMode("algorithm")}
                        >
                          <Layers3 size={13} /> Algorithm
                        </button>
                      </div>
                      <select
                        className="template-select"
                        aria-label="Load a formula template"
                        value=""
                        onChange={(e) => template(e.target.value)}
                      >
                        <option value="">↳ Load a template</option>
                        {catalog?.templates?.map((t: Data) => (
                          <option value={t.id} key={t.id}>
                            {t.name}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div className="formula-meta">
                      <label className="field">
                        <span>Formula / algorithm name</span>
                        <input
                          value={candidate.name}
                          onChange={(e) =>
                            set("candidate.name", e.target.value)
                          }
                        />
                      </label>
                      <label className="field">
                        <span>Apply to SFL hook</span>
                        <select
                          value={candidate.hook}
                          onChange={(e) =>
                            set("candidate.hook", e.target.value)
                          }
                        >
                          {catalog?.hooks?.map((h: Data) => (
                            <option key={h.id} value={h.id}>
                              {h.id} · {h.name}
                            </option>
                          ))}
                        </select>
                      </label>
                    </div>
                    {candidate.mode === "formula" ? (
                      <>
                        <div className="code-editor">
                          <div className="code-editor-top">
                            <span>
                              <span className="mini-dot" /> candidate.formula
                            </span>
                            <span>RESTRICTED EXPRESSION</span>
                          </div>
                          <div className="code-input">
                            <span className="line-number">01</span>
                            <textarea
                              ref={editor}
                              spellCheck={false}
                              aria-label="Candidate mathematical formula"
                              value={candidate.expression}
                              onChange={(e) =>
                                set("candidate.expression", e.target.value)
                              }
                              placeholder="clip(G, C_G)"
                            />
                          </div>
                          <div className="code-editor-bottom">
                            <span>
                              {hook?.input || "Tensor"} →{" "}
                              {hook?.output || "Tensor"}
                            </span>
                            <span>Hook {candidate.hook}</span>
                          </div>
                        </div>
                        <div className="math-keyboard">
                          <div
                            className="keyboard-tabs"
                            role="tablist"
                            aria-label="Mathematical keyboard categories"
                          >
                            {Object.keys(catalog?.keyboard || {}).map((k) => (
                              <button
                                role="tab"
                                aria-selected={keyboard === k}
                                key={k}
                                className={keyboard === k ? "active" : ""}
                                onClick={() => setKeyboard(k)}
                              >
                                {k === "Differential Privacy" ? "DP" : k}
                              </button>
                            ))}
                          </div>
                          <div className="keyboard-keys">
                            {(catalog?.keyboard?.[keyboard] || []).map(
                              (key: Data, i: number) => (
                                <button
                                  key={i}
                                  title={`${key.description || key.label}${key.executable === false ? " · notation only; validator will explain unsupported expressions" : ""}`}
                                  className={
                                    key.executable === false
                                      ? "notation-key"
                                      : ""
                                  }
                                  onClick={() => insert(key.insert)}
                                >
                                  {key.label}
                                </button>
                              ),
                            )}
                          </div>
                          <p className="keyboard-hint">
                            Select a symbol to insert at the cursor. Validation
                            checks the executable subset.
                          </p>
                        </div>
                      </>
                    ) : (
                      <div className="algorithm-editor">
                        <div className="algorithm-heading">
                          <span>CONTROLLED SFL OPERATIONS</span>
                          <select
                            aria-label="Add algorithm operation"
                            value=""
                            onChange={(e) => addOperation(e.target.value)}
                          >
                            <option value="">+ Add protection block</option>
                            {catalog?.operations
                              ?.filter(
                                (o: Data) =>
                                  !o.required &&
                                  !candidate.algorithm.some(
                                    (b: Data) => b.op === o.op,
                                  ),
                              )
                              .map((o: Data) => (
                                <option key={o.op} value={o.op}>
                                  {title(o.op)}
                                </option>
                              ))}
                          </select>
                        </div>
                        <div className="algorithm-blocks">
                          {candidate.algorithm.map(
                            (block: Data, index: number) => {
                              const def = catalog?.operations?.find(
                                (o: Data) => o.op === block.op,
                              );
                              return (
                                <div
                                  className={`algorithm-block ${!def?.required ? "protection" : ""}`}
                                  key={`${block.op}-${index}`}
                                >
                                  <span className="block-index">
                                    {String(index + 1).padStart(2, "0")}
                                  </span>
                                  <div>
                                    <strong title={def?.description}>
                                      {title(block.op)}
                                    </strong>
                                    {Object.entries(block.params || {}).map(
                                      ([param, v]) => (
                                        <label key={param}>
                                          {param}
                                          <input
                                            type="number"
                                            step="any"
                                            value={String(v)}
                                            aria-label={`${block.op} ${param}`}
                                            onChange={(e) => {
                                              const blocks = structuredClone(
                                                candidate.algorithm,
                                              );
                                              blocks[index].params[param] =
                                                Number(e.target.value);
                                              set(
                                                "candidate.algorithm",
                                                blocks,
                                              );
                                            }}
                                          />
                                        </label>
                                      ),
                                    )}
                                  </div>
                                  <span className="block-type">
                                    {def?.required ? "CORE" : "HOOK"}
                                  </span>
                                  {!def?.required && (
                                    <button
                                      className="icon-button"
                                      aria-label={`Remove ${block.op}`}
                                      onClick={() =>
                                        set(
                                          "candidate.algorithm",
                                          candidate.algorithm.filter(
                                            (_: any, i: number) => i !== index,
                                          ),
                                        )
                                      }
                                    >
                                      <Trash2 size={13} />
                                    </button>
                                  )}
                                  {index > 0 &&
                                    def?.stage ===
                                      catalog?.operations?.find(
                                        (o: Data) =>
                                          o.op ===
                                          candidate.algorithm[index - 1].op,
                                      )?.stage && (
                                      <button
                                        className="icon-button"
                                        aria-label={`Move ${block.op} earlier`}
                                        onClick={() => moveOperation(index, -1)}
                                      >
                                        <ArrowUp size={13} />
                                      </button>
                                    )}
                                  {index < candidate.algorithm.length - 1 &&
                                    def?.stage ===
                                      catalog?.operations?.find(
                                        (o: Data) =>
                                          o.op ===
                                          candidate.algorithm[index + 1].op,
                                      )?.stage && (
                                      <button
                                        className="icon-button"
                                        aria-label={`Move ${block.op} later`}
                                        onClick={() => moveOperation(index, 1)}
                                      >
                                        <ArrowDown size={13} />
                                      </button>
                                    )}
                                </div>
                              );
                            },
                          )}
                        </div>
                        <p className="caption">
                          Blocks are inserted at their permitted stage. Core
                          transport, loss, backward, and aggregation stages are
                          required.
                        </p>
                      </div>
                    )}
                    <div className="parameter-row">
                      <span className="parameter-label">PARAMETERS</span>
                      {Object.entries(candidate.parameters || {}).map(
                        ([name, value]) => (
                          <label className="parameter-input" key={name}>
                            <span>{name}</span>
                            <input
                              type="number"
                              step="any"
                              value={String(value)}
                              aria-label={`Parameter ${name}`}
                              onChange={(e) =>
                                set(
                                  `candidate.parameters.${name}`,
                                  Number(e.target.value),
                                )
                              }
                            />
                            <button
                              aria-label={`Remove parameter ${name}`}
                              onClick={() => {
                                const next = { ...candidate.parameters };
                                delete next[name];
                                set("candidate.parameters", next);
                              }}
                            >
                              <X size={11} />
                            </button>
                          </label>
                        ),
                      )}
                      <div className="add-parameter">
                        <input
                          placeholder="Parameter"
                          aria-label="New parameter name"
                          value={parameterName}
                          onChange={(e) => setParameterName(e.target.value)}
                        />
                        <button
                          title="Add numeric parameter"
                          aria-label="Add numeric parameter"
                          onClick={() => {
                            if (
                              /^[A-Za-z_][A-Za-z0-9_]*$/.test(parameterName)
                            ) {
                              set(`candidate.parameters.${parameterName}`, 1);
                              setParameterName("");
                            }
                          }}
                        >
                          <Plus size={13} />
                        </button>
                      </div>
                    </div>
                    <details className="formula-information">
                      <summary>
                        <CircleHelp size={13} /> Formula information{" "}
                        <ChevronDown size={13} />
                      </summary>
                      <dl className="definition-list">
                        <div>
                          <dt>Selected hook</dt>
                          <dd>
                            {hook?.id} · {hook?.name}
                          </dd>
                        </div>
                        <div>
                          <dt>Purpose</dt>
                          <dd>{hook?.description}</dd>
                        </div>
                        <div>
                          <dt>Input</dt>
                          <dd>{hook?.input}</dd>
                        </div>
                        <div>
                          <dt>Output</dt>
                          <dd>{hook?.output}</dd>
                        </div>
                        <div>
                          <dt>Available variables</dt>
                          <dd>{hook?.variables?.join(", ")}</dd>
                        </div>
                        {validation?.formula_info &&
                          Object.entries(validation.formula_info)
                            .filter(([k]) => /property|category/.test(k))
                            .map(([k, v]) => (
                              <div key={k}>
                                <dt>{title(k)}</dt>
                                <dd>{String(v)}</dd>
                              </div>
                            ))}
                      </dl>
                    </details>
                    <label className="hypothesis field">
                      <span>
                        <Beaker size={13} /> Research hypothesis{" "}
                        <span className="required-label">REQUIRED</span>
                      </span>
                      <textarea
                        value={config.hypothesis}
                        onChange={(e) => set("hypothesis", e.target.value)}
                        rows={3}
                        placeholder="What should improve, under which constraints?"
                      />
                      <small>
                        Define your expected effect before running. Acceptance
                        criteria are evaluated against measured results.
                      </small>
                    </label>
                  </div>
                  <div className="editor-actions">
                    <button
                      className="button secondary"
                      onClick={() => void validate()}
                      disabled={!!busy || active}
                    >
                      {busy === "validate" ? (
                        <LoaderCircle size={14} className="spin" />
                      ) : (
                        <ShieldCheck size={14} />
                      )}{" "}
                      Validate candidate
                    </button>
                    <span>
                      {validation ? (
                        <Badge status={validation.valid ? "valid" : "invalid"}>
                          {validation.valid
                            ? "Ready for experiment"
                            : "Changes required"}
                        </Badge>
                      ) : (
                        <span className="validation-pending">
                          <i /> Awaiting validation
                        </span>
                      )}
                    </span>
                  </div>
                  {validation && (
                    <div className="validation-results">
                      <Checks checks={validation.checks || []} />
                    </div>
                  )}
                </section>
                <section className="pipeline-panel panel">
                  <div className="section-heading">
                    <div className="heading-with-icon">
                      <Network size={16} />
                      <h3>Your SFL pipeline</h3>
                    </div>
                    <span className="tiny-label">
                      U-SHAPED · {config.split.join("–")}
                    </span>
                  </div>
                  <SplitDiagram split={config.split} active={active} />
                  <div className="pipeline-foot">
                    <span>
                      <ShieldCheck size={13} /> Labels stay on the client
                    </span>
                    <span>
                      <Database size={13} /> Logical client boundaries
                    </span>
                  </div>
                </section>
                <section className="launch-panel">
                  <div>
                    <span className="eyebrow">READY TO INVESTIGATE?</span>
                    <h3>Let the experiment decide.</h3>
                    <p>Compare controlled runs. Evaluate trade-offs.</p>
                  </div>
                  <div className="launch-actions">
                    <div>
                      <button
                        className="button secondary"
                        onClick={() => void run("baseline")}
                        disabled={!!busy || active}
                      >
                        <Play size={12} /> Baseline
                      </button>
                      <button
                        className="button secondary"
                        onClick={() => void run("candidate")}
                        disabled={!!busy || active}
                      >
                        <Play size={12} /> Candidate
                      </button>
                      <button
                        className="button secondary"
                        onClick={() => void run("compare")}
                        disabled={!!busy || active}
                      >
                        <Layers3 size={12} /> Compare
                      </button>
                    </div>
                    <button
                      className="button primary full-evaluation"
                      onClick={() => void run("full")}
                      disabled={!!busy || active}
                    >
                      {busy && busy !== "validate" ? (
                        <LoaderCircle size={15} className="spin" />
                      ) : (
                        <FlaskConical size={16} />
                      )}{" "}
                      Run full evaluation <ArrowUpRight size={16} />
                    </button>
                  </div>
                </section>
              </div>
            </div>
          )}
          <section className="results-section" ref={resultsRef}>
            <div className="results-heading">
              <div>
                <div className="eyebrow">OBSERVE · COMPARE · CONCLUDE</div>
                <h2>Results & analysis</h2>
              </div>
              <div>
                {selected ? (
                  <>
                    <span className="experiment-id">{selected.id}</span>
                    <Badge status={selected.status} />
                    <button
                      className="button secondary small"
                      onClick={() => setEvidenceOpen(true)}
                    >
                      <EyeIcon /> View evidence
                    </button>
                  </>
                ) : (
                  <span className="badge muted">No experiment selected</span>
                )}
              </div>
            </div>
            {selected && (
              <section className="progress-panel panel">
                <div className="progress-title">
                  <div>
                    <span
                      className={active ? "status-dot pulse" : "status-dot"}
                    />
                    <strong>{selected.name}</strong>
                    <span>{title(selected.phase || selected.status)}</span>
                  </div>
                  <div>
                    <span>{fmt(selected.progress || 0, 0)}%</span>
                    {active && (
                      <button
                        className="button danger small"
                        disabled={busy === "cancel"}
                        onClick={() => void cancel()}
                      >
                        <Square size={10} /> Cancel
                      </button>
                    )}
                  </div>
                </div>
                <div className="progress-bar">
                  <span
                    style={{
                      width: `${Math.min(100, selected.progress || 0)}%`,
                    }}
                  />
                </div>
                <div className="progress-stages">
                  {stages.map((s, i) => (
                    <div
                      key={s}
                      className={
                        selected.status === "completed"
                          ? "done"
                          : i < phaseIndex(selected.phase)
                            ? "done"
                            : i === phaseIndex(selected.phase) && active
                              ? "current"
                              : ""
                      }
                    >
                      <span>
                        {selected.status === "completed" ||
                        i < phaseIndex(selected.phase) ? (
                          <Check size={11} />
                        ) : (
                          i + 1
                        )}
                      </span>
                      {s}
                    </div>
                  ))}
                </div>
                {selected.error && (
                  <div className="job-error" role="alert">
                    {typeof selected.error === "string"
                      ? selected.error
                      : JSON.stringify(selected.error)}
                  </div>
                )}
                <details className="job-log">
                  <summary>
                    <Terminal size={13} /> Execution log{" "}
                    <ChevronDown size={12} />
                  </summary>
                  <pre>
                    {(selected.logs || [])
                      .slice(-60)
                      .map((l) =>
                        typeof l === "string" ? l : JSON.stringify(l),
                      )
                      .join("\n") || "Waiting for execution events…"}
                  </pre>
                </details>
              </section>
            )}
            <div
              className="result-tabs"
              role="tablist"
              aria-label="Experiment result views"
            >
              {tabs.map((t) => (
                <button
                  role="tab"
                  aria-selected={tab === t}
                  key={t}
                  className={tab === t ? "active" : ""}
                  onClick={() => setTab(t)}
                >
                  {t}
                </button>
              ))}
            </div>
            <div className="result-content" role="tabpanel" aria-label={tab}>
              <Results
                experiment={selected}
                tab={tab}
                dark={dark}
                onEvidence={() => setEvidenceOpen(true)}
              />
            </div>
          </section>
          <footer className="page-footer">
            <span>
              <span className="brand-mini">
                <Layers3 size={14} />
              </span>{" "}
              SFL-Lab · Formula & Algorithm Validation Workbench
            </span>
            <span>Measured outcomes. Explicit assumptions.</span>
          </footer>
        </main>
      </div>
      {historyOpen && (
        <div className="drawer-overlay" onClick={() => setHistoryOpen(false)}>
          <aside
            className="drawer history-drawer"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="drawer-heading">
              <div>
                <span className="eyebrow">YOUR RESEARCH TRAIL</span>
                <h2>Experiment history</h2>
              </div>
              <button
                className="icon-button"
                aria-label="Close history"
                onClick={() => setHistoryOpen(false)}
              >
                <X size={20} />
              </button>
            </div>
            <p className="caption">
              Select a saved experiment to inspect its results and
              reproducibility evidence.
            </p>
            {experiments.length ? (
              experiments.map((e) => (
                <button
                  className="history-item"
                  key={e.id}
                  onClick={() => void selectExperiment(e)}
                >
                  <div>
                    <span className="experiment-id">{e.id}</span>
                    <Badge status={e.status} />
                  </div>
                  <h3>{e.name}</h3>
                  <p>
                    {new Date(e.created_at).toLocaleString()}
                    <ArrowUpRight size={15} />
                  </p>
                </button>
              ))
            ) : (
              <div className="drawer-empty">
                <History size={29} />
                <h3>Your research trail starts here</h3>
                <p>Experiments are saved automatically when you run them.</p>
              </div>
            )}
          </aside>
        </div>
      )}
      {evidenceOpen && selected && (
        <div className="drawer-overlay" onClick={() => setEvidenceOpen(false)}>
          <aside
            className="drawer evidence-drawer"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="drawer-heading">
              <div>
                <span className="eyebrow">REPRODUCIBILITY RECORD</span>
                <h2>Experiment evidence</h2>
              </div>
              <button
                className="icon-button"
                aria-label="Close evidence"
                onClick={() => setEvidenceOpen(false)}
              >
                <X size={20} />
              </button>
            </div>
            <p>
              {selected.name}{" "}
              <span className="experiment-id">{selected.id}</span>
            </p>
            <div className="evidence-tabs">
              {["Configuration", "Results", "Logs", "Environment"].map((t) => (
                <button
                  key={t}
                  className={evidenceTab === t ? "active" : ""}
                  onClick={() => setEvidenceTab(t)}
                >
                  {t}
                </button>
              ))}
            </div>
            <div className="evidence-actions">
              <button
                className="button secondary small"
                onClick={() =>
                  download(
                    JSON.stringify(evidence, null, 2),
                    `${selected.id}-${evidenceTab.toLowerCase()}.json`,
                    "application/json",
                  )
                }
              >
                <FileText size={13} /> Download JSON
              </button>
              {evidenceTab === "Configuration" && (
                <button
                  className="button secondary small"
                  onClick={() => {
                    setConfig(structuredClone(selected.config));
                    setValidation(null);
                    setEvidenceOpen(false);
                    setupRef.current?.scrollIntoView({ behavior: "smooth" });
                  }}
                >
                  <Copy size={13} /> Use configuration
                </button>
              )}
            </div>
            <pre className="evidence-json">
              {JSON.stringify(evidence, null, 2)}
            </pre>
          </aside>
        </div>
      )}
      {helpOpen && (
        <div className="drawer-overlay" onClick={() => setHelpOpen(false)}>
          <aside
            className="drawer help-drawer"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="drawer-heading">
              <div>
                <span className="eyebrow">SCIENTIFIC METHODOLOGY</span>
                <h2>A controlled research workflow</h2>
              </div>
              <button
                className="icon-button"
                aria-label="Close methodology"
                onClick={() => setHelpOpen(false)}
              >
                <X size={20} />
              </button>
            </div>
            <div className="help-content">
              <h3>1. State an effect you can test</h3>
              <p>
                Choose a formula or controlled algorithm, its exact SFL hook, a
                hypothesis, and numerical acceptance criteria. Validation checks
                supported syntax, variables, domains, and operation order.
              </p>
              <h3>2. Keep the comparison controlled</h3>
              <p>
                Baseline and candidate use matched data partitions, seeds, model
                initialization, and training settings. The U-shaped pipeline
                keeps labels and the loss on the logical client. Activations and
                split gradients cross explicit boundaries.
              </p>
              <h3>3. Protect the final test set</h3>
              <p>
                Training, validation, and test sets are separated. Validation
                selects checkpoints and sweep parameters. Exploratory tests and
                small seed counts limit conclusions.
              </p>
              <h3>4. Distinguish evidence from estimates</h3>
              <p>
                Accuracy, tensor bytes, and execution runtime come from executed
                runs. Network latency and energy are model estimates.
                Client/server distribution is simulated locally; it does not
                demonstrate real distributed speedup.
              </p>
              <h3>5. Interpret privacy within its scope</h3>
              <p>
                Clipping and noise may protect specific tensor releases. The
                conservative RDP accountant covers the configured transcript; it
                does not establish full-system or final-model privacy.
                Reconstruction probes test scoped attacks, not universal
                resistance.
              </p>
              <h3>6. Draw a constrained conclusion</h3>
              <p>
                The recommendation uses acceptance criteria, architecture
                compliance, statistical uncertainty, and trade-offs. An unrun
                suite, missing metric, or unsupported claim stays explicitly
                unresolved. Results cannot establish generalization to datasets
                or models that were not tested.
              </p>
              <div className="note">
                <ShieldCheck size={18} />
                <span>
                  OS-USFL-inspired reference architecture. Original paper
                  reproduction requires the original SeaShips dataset and
                  verified original experimental settings.
                </span>
              </div>
            </div>
          </aside>
        </div>
      )}
    </div>
  );
}
function EyeIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 20 20" fill="none">
      <path
        d="M2 10s3-5 8-5 8 5 8 5-3 5-8 5-8-5-8-5Z"
        stroke="currentColor"
        strokeWidth="1.5"
      />
      <circle cx="10" cy="10" r="2" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}
