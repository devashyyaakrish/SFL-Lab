export type Data = Record<string, any>;
export interface Experiment {
  id: string;
  name: string;
  status: string;
  phase: string;
  progress: number;
  config: Data;
  created_at: string;
  updated_at: string;
  error?: string | null;
  result?: Data | null;
  logs: any[];
}
export const tabs = [
  "Overview",
  "Architecture",
  "Learning",
  "Privacy",
  "Federation",
  "Communication",
  "Scalability",
  "Network",
  "Parameters",
  "Ablation",
  "Statistics",
  "Attacks",
  "Final Report",
];
export const colors = {
  baseline: "#9d99ef",
  candidate: "#67d9bf",
  amber: "#efbb71",
  red: "#f18792",
  blue: "#77b5f5",
};
export const metricNames: Data = {
  test_accuracy: "Test accuracy",
  val_accuracy: "Validation accuracy",
  train_accuracy: "Train accuracy",
  test_loss: "Test loss",
  val_loss: "Validation loss",
  train_loss: "Train loss",
  runtime_seconds: "Runtime",
  communication_bytes: "Communication",
  energy_joules: "Energy estimate",
  epsilon: "Privacy ε",
  drift_mean: "Mean client drift",
  fairness_std: "Accuracy disparity",
  latency_seconds: "Latency estimate",
  network_seconds: "Network time estimate",
  aggregation_seconds: "Aggregation time",
};
export const metricUnits: Data = {
  test_accuracy: "%",
  val_accuracy: "%",
  train_accuracy: "%",
  test_loss: "loss",
  val_loss: "loss",
  train_loss: "loss",
  runtime_seconds: "s",
  communication_bytes: "bytes",
  energy_joules: "J",
  epsilon: "ε",
  drift_mean: "L2",
  fairness_std: "pp",
  latency_seconds: "s",
  network_seconds: "s",
  aggregation_seconds: "s",
};
export function numeric(v: any): v is number {
  return typeof v === "number" && Number.isFinite(v);
}
export function fmt(v: any, digits = 2): string {
  return numeric(v)
    ? v.toLocaleString(undefined, { maximumFractionDigits: digits })
    : "—";
}
export function bytes(v: any): string {
  if (!numeric(v)) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let n = v,
    i = 0;
  while (Math.abs(n) >= 1000 && i < 3) {
    n /= 1000;
    i++;
  }
  return `${fmt(n)} ${units[i]}`;
}
export function metric(v: any, key: string): string {
  if (key === "communication_bytes") return bytes(v);
  return `${fmt(v)}${numeric(v) ? " " + (metricUnits[key] || "") : ""}`.trim();
}
export function title(v: any): string {
  return String(v ?? "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (s) => s.toUpperCase());
}
export function download(
  content: string,
  filename: string,
  type = "text/plain",
) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export async function api(path: string, options?: RequestInit) {
  const response = await fetch("/api" + path, options);
  if (!response.ok) {
    const e = await response
      .json()
      .catch(() => ({ detail: response.statusText }));
    throw new Error(
      typeof e.detail === "string" ? e.detail : JSON.stringify(e.detail || e),
    );
  }
  return response.json();
}
export function jsonOptions(body: any): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}
