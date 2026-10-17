import { useEffect, useRef } from "react";
import * as echarts from "echarts/core";
import {
  LineChart,
  BarChart,
  ScatterChart,
  BoxplotChart,
  HeatmapChart,
  SankeyChart,
} from "echarts/charts";
import {
  GridComponent,
  LegendComponent,
  TooltipComponent,
  VisualMapComponent,
  MarkLineComponent,
  MarkAreaComponent,
} from "echarts/components";
import { CanvasRenderer, SVGRenderer } from "echarts/renderers";
import { Download, ChartNoAxesCombined } from "lucide-react";
import { colors, numeric } from "./types";
import type { Data } from "./types";
echarts.use([
  LineChart,
  BarChart,
  ScatterChart,
  BoxplotChart,
  HeatmapChart,
  SankeyChart,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  VisualMapComponent,
  MarkLineComponent,
  MarkAreaComponent,
  CanvasRenderer,
  SVGRenderer,
]);
export function Empty({
  title = "Evidence will appear here",
  detail = "Run an experiment to generate measured results for this view.",
}: {
  title?: string;
  detail?: string;
}) {
  return (
    <div className="empty">
      <span className="empty-icon">
        <ChartNoAxesCombined size={29} />
      </span>
      <h3>{title}</h3>
      <p>{detail}</p>
    </div>
  );
}
export function Chart({
  title,
  subtitle,
  option,
  height = 280,
  dark = true,
}: {
  title: string;
  subtitle?: string;
  option: Data | null;
  height?: number;
  dark?: boolean;
}) {
  const element = useRef<HTMLDivElement>(null),
    chart = useRef<echarts.ECharts | null>(null),
    fullOption = useRef<Data>({});
  useEffect(() => {
    if (!element.current || !option) return;
    const instance = echarts.init(element.current, undefined, {
      renderer: "canvas",
    });
    chart.current = instance;
    const foreground = dark ? "#94a1b6" : "#657187",
      gridColor = dark ? "#27303f" : "#e8edf3";
    fullOption.current = {
      animationDuration: 400,
      color: [
        colors.baseline,
        colors.candidate,
        colors.amber,
        colors.blue,
        colors.red,
      ],
      textStyle: {
        fontFamily: "Inter, system-ui, sans-serif",
        color: foreground,
      },
      tooltip: {
        trigger: "axis",
        backgroundColor: dark ? "#1b2431" : "#fff",
        borderColor: gridColor,
        textStyle: { color: dark ? "#e3e9f1" : "#1d293a" },
        confine: true,
      },
      legend: {
        bottom: 0,
        textStyle: { color: foreground },
        itemWidth: 14,
        itemHeight: 3,
      },
      grid: { left: 58, right: 22, top: 30, bottom: 52, containLabel: false },
      ...option,
    };
    const axes = (axis: any): any =>
      Array.isArray(axis)
        ? axis.map(axes)
        : axis
          ? {
              axisLine: { lineStyle: { color: gridColor } },
              axisTick: { show: false },
              axisLabel: { color: foreground, fontSize: 10 },
              splitLine: { lineStyle: { color: gridColor, type: "dashed" } },
              nameTextStyle: { color: foreground, fontSize: 10 },
              ...axis,
            }
          : undefined;
    if (fullOption.current.xAxis)
      fullOption.current.xAxis = axes(fullOption.current.xAxis);
    if (fullOption.current.yAxis)
      fullOption.current.yAxis = axes(fullOption.current.yAxis);
    instance.setOption(fullOption.current);
    const observer = new ResizeObserver(() => instance.resize());
    observer.observe(element.current);
    return () => {
      observer.disconnect();
      instance.dispose();
      chart.current = null;
    };
  }, [option, dark]);
  function save(format: "png" | "svg") {
    if (!chart.current) return;
    let url: string;
    if (format === "svg") {
      const div = document.createElement("div");
      const clone = echarts.init(div, undefined, {
        renderer: "svg",
        width: 900,
        height: 430,
      });
      clone.setOption({
        ...fullOption.current,
        animation: false,
        backgroundColor: dark ? "#151c27" : "#fff",
      });
      url = clone.getDataURL({ type: "svg" });
      clone.dispose();
    } else
      url = chart.current.getDataURL({
        type: "png",
        pixelRatio: 2,
        backgroundColor: dark ? "#151c27" : "#fff",
      });
    const a = document.createElement("a");
    a.href = url;
    a.download = title.toLowerCase().replaceAll(" ", "-") + "." + format;
    a.click();
  }
  return (
    <section className="chart-card panel">
      <div className="section-heading">
        <div>
          <h3>{title}</h3>
          {subtitle && <p>{subtitle}</p>}
        </div>
        {option && (
          <div className="chart-exports">
            <button onClick={() => save("png")} title="Export chart as PNG">
              <Download size={12} /> PNG
            </button>
            <button onClick={() => save("svg")} title="Export chart as SVG">
              SVG
            </button>
          </div>
        )}
      </div>
      {option ? (
        <div
          className="chart"
          ref={element}
          role="img"
          aria-label={title}
          style={{ height }}
        />
      ) : (
        <Empty title="No measurements yet" />
      )}
    </section>
  );
}
export function meanHistory(runs: Data[], key: string) {
  const rounds = [
    ...new Set(
      runs.flatMap((r) => (r.history || []).map((h: Data) => h.round)),
    ),
  ].sort((a: any, b: any) => a - b);
  return rounds.map((round) => {
    const vals = runs
      .flatMap((r) =>
        (r.history || [])
          .filter((h: Data) => h.round === round)
          .map((h: Data) => h[key]),
      )
      .filter(numeric);
    return [
      round,
      vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null,
    ];
  });
}
export function lineOption(
  baseline: Data[],
  candidate: Data[],
  key: string,
  unit: string,
  fixedAccuracy = false,
): Data | null {
  const b = meanHistory(baseline, key),
    c = meanHistory(candidate, key);
  if (![...b, ...c].some((p) => numeric(p[1]))) return null;
  return {
    xAxis: { type: "value", name: "Round", minInterval: 1 },
    yAxis: {
      type: "value",
      name: unit,
      ...(fixedAccuracy ? { min: 0, max: 100 } : {}),
    },
    series: [
      {
        name: "Baseline",
        type: "line",
        smooth: false,
        symbolSize: 6,
        lineStyle: { width: 2 },
        data: b,
      },
      {
        name: "Candidate",
        type: "line",
        smooth: false,
        symbolSize: 6,
        lineStyle: { width: 2 },
        data: c,
      },
    ],
  };
}
