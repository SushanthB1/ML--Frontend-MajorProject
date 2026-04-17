import React, { useState, useRef, useCallback, useMemo } from "react";
import {
  UploadCloud,
  Database,
  Play,
  CheckCircle,
  AlertCircle,
  BarChart2,
  Trophy,
  FunctionSquare,
  Layers,
  Target,
  Cpu,
  Sliders,
  ChevronRight,
  TrendingUp,
  TrendingDown,
  Zap,
  FlaskConical,
  Microscope,
  Mountain,
  BookOpen,
  Eye,
  RefreshCw,
  Info,
  Download,
} from "lucide-react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// =====================================================================
// TYPES
// =====================================================================
type Tab = "upload" | "train" | "predict";

interface ChartDataPoint {
  actual: number;
  predicted: number;
}
interface MetricSet {
  mse: number | null;
  rmse: number | null;
  mae: number | null;
  r2: number | null;
}
interface ModelMetrics {
  train: MetricSet;
  test: MetricSet;
  mse: number | null;
  r2: number | null;
}
interface ShapItem {
  feature: string;
  importance: number;
}
interface ComparisonData {
  [modelName: string]: ModelMetrics;
}
interface ColumnStats {
  count: number;
  mean: number | null;
  std: number | null;
  min: number | null;
  q25: number | null;
  median: number | null;
  q75: number | null;
  max: number | null;
  missing: number;
  missing_pct: number | null;
}
interface TrainResponse {
  comparison: ComparisonData;
  best_model: string;
  reasoning: string;
  pysr_equation: string;
  chart_data: { [modelName: string]: ChartDataPoint[] };
  shap_data: { [modelName: string]: ShapItem[] };
  resolved_features: string[];
  resolved_target: string;
  trained_models: string[];
  split_info: { train_size: number; test_size: number; test_ratio: number };
  training_warnings?: { [k: string]: string };
  hyperparameters?: { [modelName: string]: Record<string, any> };
}

// Ensure XGBoost is explicitly in the models array
const ALL_MODELS = [
  "Random Forest",
  "XGBoost",
  "SVR",
  "ORF",
  "PySR (Symbolic)",
];

// =====================================================================
// ROCK MECHANICS UNITS MAP
// =====================================================================
const ROCK_MECHANICS_UNITS: [RegExp, string][] = [
  [/\bucs\b/i, "MPa"],
  [/unconfined.compressive.strength/i, "MPa"],
  [/compressive.strength/i, "MPa"],
  [/confining.pressure/i, "MPa"],
  [/\bsigma.?3\b/i, "MPa"],
  [/\bσ3\b/, "MPa"],
  [/tensile.strength/i, "MPa"],
  [/\bbts\b/i, "MPa"],
  [/point.load/i, "MPa"],
  [/\bis50\b/i, "MPa"],
  [/cohesion/i, "MPa"],
  [/shear.strength/i, "MPa"],
  [/deviator/i, "MPa"],
  [/triaxial/i, "MPa"],
  [/young.?s.modulus/i, "GPa"],
  [/elastic.modulus/i, "GPa"],
  [/bulk.modulus/i, "GPa"],
  [/shear.modulus/i, "GPa"],
  [/friction.angle/i, "°"],
  [/internal.friction/i, "°"],
  [/\bphi\b/i, "°"],
  [/dip.angle/i, "°"],
  [/p.wave.velocity/i, "km/s"],
  [/s.wave.velocity/i, "km/s"],
  [/\bvp\b/i, "km/s"],
  [/\bvs\b/i, "km/s"],
  [/wave.velocity/i, "km/s"],
  [/\bdensity\b/i, "g/cm³"],
  [/unit.weight/i, "kN/m³"],
  [/porosity/i, "%"],
  [/water.content/i, "%"],
  [/moisture.content/i, "%"],
  [/saturation/i, "%"],
  [/void.ratio/i, ""],
  [/poisson.?s.ratio/i, ""],
  [/\bRQD\b/, "%"],
  [/schmidt.hammer/i, ""],
  [/rebound/i, ""],
  [/hardness/i, ""],
  [/slake/i, ""],
  [/depth/i, "m"],
  [/thickness/i, "m"],
  [/length/i, "mm"],
  [/diameter/i, "mm"],
  [/height/i, "mm"],
];

function getUnit(colName: string): string {
  for (const [pattern, unit] of ROCK_MECHANICS_UNITS) {
    if (pattern.test(colName)) return unit;
  }
  const parenMatch = colName.match(/\(([^)]+)\)\s*$/);
  if (parenMatch) return parenMatch[1];
  return "";
}

// =====================================================================
// RESOURCES
// =====================================================================
const ROCK_IMAGES = [
  {
    url: "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=400&h=280&fit=crop",
    label: "Core Samples",
    icon: <Microscope className="w-4 h-4" />,
  },
  {
    url: "https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=400&h=280&fit=crop",
    label: "Rock Specimens",
    icon: <Mountain className="w-4 h-4" />,
  },
  {
    url: "https://images.unsplash.com/photo-1509316785289-025f5b846b35?w=400&h=280&fit=crop",
    label: "Geological Strata",
    icon: <Layers className="w-4 h-4" />,
  },
  {
    url: "https://images.unsplash.com/photo-1584515979956-d9f6e5d09982?w=400&h=280&fit=crop",
    label: "Rock Testing",
    icon: <FlaskConical className="w-4 h-4" />,
  },
];

const UCSTestDiagram = () => (
  <svg viewBox="0 0 120 160" className="w-full h-full">
    <rect x="20" y="5" width="80" height="10" rx="2" fill="#475569" />
    <rect x="45" y="15" width="30" height="8" rx="1" fill="#64748b" />
    <rect
      x="40"
      y="23"
      width="40"
      height="80"
      rx="4"
      fill="#c2a982"
      stroke="#a08060"
      strokeWidth="1.5"
    />
    <line
      x1="40"
      y1="35"
      x2="80"
      y2="35"
      stroke="#a08060"
      strokeOpacity="0.5"
      strokeWidth="0.5"
    />
    <line
      x1="40"
      y1="50"
      x2="80"
      y2="50"
      stroke="#a08060"
      strokeOpacity="0.5"
      strokeWidth="0.5"
    />
    <line
      x1="40"
      y1="65"
      x2="80"
      y2="65"
      stroke="#a08060"
      strokeOpacity="0.5"
      strokeWidth="0.5"
    />
    <line
      x1="40"
      y1="80"
      x2="80"
      y2="80"
      stroke="#a08060"
      strokeOpacity="0.5"
      strokeWidth="0.5"
    />
    <rect x="45" y="103" width="30" height="8" rx="1" fill="#64748b" />
    <rect x="20" y="111" width="80" height="10" rx="2" fill="#475569" />
    <line
      x1="60"
      y1="0"
      x2="60"
      y2="5"
      stroke="#ef4444"
      strokeWidth="2"
      markerEnd="url(#arrow)"
    />
    <line x1="60" y1="121" x2="60" y2="126" stroke="#ef4444" strokeWidth="2" />
    <text
      x="60"
      y="145"
      textAnchor="middle"
      fontSize="9"
      fill="#64748b"
      fontFamily="monospace"
    >
      UCS Test
    </text>
    <text
      x="60"
      y="155"
      textAnchor="middle"
      fontSize="7"
      fill="#94a3b8"
      fontFamily="monospace"
    >
      Axial Load
    </text>
  </svg>
);
const TriaxialDiagram = () => (
  <svg viewBox="0 0 140 160" className="w-full h-full">
    <rect x="30" y="10" width="80" height="8" rx="2" fill="#475569" />
    <rect
      x="20"
      y="22"
      width="100"
      height="100"
      rx="6"
      fill="none"
      stroke="#3b82f6"
      strokeWidth="2"
      strokeDasharray="4 2"
    />
    <rect
      x="50"
      y="25"
      width="40"
      height="90"
      rx="4"
      fill="#c2a982"
      stroke="#a08060"
      strokeWidth="1.5"
    />
    <rect x="30" y="115" width="80" height="8" rx="2" fill="#475569" />
    <line x1="20" y1="72" x2="50" y2="72" stroke="#f59e0b" strokeWidth="2" />
    <line x1="90" y1="72" x2="120" y2="72" stroke="#f59e0b" strokeWidth="2" />
    <text x="70" y="10" textAnchor="middle" fontSize="7" fill="#3b82f6">
      σ₃ (Confining)
    </text>
    <text
      x="70"
      y="145"
      textAnchor="middle"
      fontSize="9"
      fill="#64748b"
      fontFamily="monospace"
    >
      Triaxial Test
    </text>
    <text
      x="70"
      y="155"
      textAnchor="middle"
      fontSize="7"
      fill="#94a3b8"
      fontFamily="monospace"
    >
      σ₁ + σ₃ Applied
    </text>
  </svg>
);
const BrazilianDiagram = () => (
  <svg viewBox="0 0 140 160" className="w-full h-full">
    <ellipse
      cx="70"
      cy="75"
      rx="50"
      ry="40"
      fill="#c2a982"
      stroke="#a08060"
      strokeWidth="1.5"
    />
    <line x1="70" y1="10" x2="70" y2="35" stroke="#ef4444" strokeWidth="2" />
    <line x1="70" y1="115" x2="70" y2="140" stroke="#ef4444" strokeWidth="2" />
    <line
      x1="30"
      y1="75"
      x2="50"
      y2="75"
      stroke="#64748b"
      strokeWidth="1"
      strokeDasharray="3 2"
    />
    <line
      x1="90"
      y1="75"
      x2="110"
      y2="75"
      stroke="#64748b"
      strokeWidth="1"
      strokeDasharray="3 2"
    />
    <text
      x="70"
      y="155"
      textAnchor="middle"
      fontSize="9"
      fill="#64748b"
      fontFamily="monospace"
    >
      Brazilian Test
    </text>
  </svg>
);
const PointLoadDiagram = () => (
  <svg viewBox="0 0 140 160" className="w-full h-full">
    <polygon points="70,15 50,35 90,35" fill="#475569" />
    <polygon points="70,135 50,115 90,115" fill="#475569" />
    <ellipse
      cx="70"
      cy="75"
      rx="35"
      ry="45"
      fill="#c2a982"
      stroke="#a08060"
      strokeWidth="1.5"
    />
    <line x1="70" y1="35" x2="70" y2="45" stroke="#ef4444" strokeWidth="1.5" />
    <line
      x1="70"
      y1="105"
      x2="70"
      y2="115"
      stroke="#ef4444"
      strokeWidth="1.5"
    />
    <text
      x="70"
      y="155"
      textAnchor="middle"
      fontSize="9"
      fill="#64748b"
      fontFamily="monospace"
    >
      Point Load Test
    </text>
  </svg>
);

const DIAGRAMS = [
  {
    component: <UCSTestDiagram />,
    label: "Uniaxial Compressive Strength",
    abbr: "UCS",
  },
  {
    component: <TriaxialDiagram />,
    label: "Triaxial Compression Test",
    abbr: "TXL",
  },
  {
    component: <BrazilianDiagram />,
    label: "Brazilian Tensile Strength",
    abbr: "BTS",
  },
  {
    component: <PointLoadDiagram />,
    label: "Point Load Strength Index",
    abbr: "PLT",
  },
];

// =====================================================================
// CHARTS
// =====================================================================
const ScatterPlot = ({
  data,
  color = "#6366f1",
  targetLabel = "Value",
}: {
  data: ChartDataPoint[];
  color?: string;
  targetLabel?: string;
}) => {
  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-stone-400 text-sm font-medium">
        No scatter data available for this model.
      </div>
    );
  }

  const pad = { top: 30, right: 20, bottom: 50, left: 55 };
  const W = 560,
    H = 320;
  const inner = { w: W - pad.left - pad.right, h: H - pad.top - pad.bottom };

  const vals = [
    ...data.map((d) => d.actual),
    ...data.map((d) => d.predicted),
  ].filter((v) => v !== null && !isNaN(v));
  if (vals.length === 0)
    return (
      <div className="flex items-center justify-center h-full text-stone-400">
        Invalid prediction data.
      </div>
    );

  const minV = Math.min(...vals) * 0.95,
    maxV = Math.max(...vals) * 1.05;
  const range = maxV - minV || 1;
  const toX = (v: number) => pad.left + ((v - minV) / range) * inner.w;
  const toY = (v: number) => pad.top + (1 - (v - minV) / range) * inner.h;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => minV + range * t);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full">
      <defs>
        <linearGradient id="gridLine" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#e2e8f0" stopOpacity="0.3" />
          <stop offset="100%" stopColor="#e2e8f0" stopOpacity="0.8" />
        </linearGradient>
      </defs>
      {ticks.map((v, i) => (
        <g key={i}>
          <line
            x1={pad.left}
            y1={toY(v)}
            x2={W - pad.right}
            y2={toY(v)}
            stroke="#e2e8f0"
            strokeWidth="1"
          />
          <line
            x1={toX(v)}
            y1={pad.top}
            x2={toX(v)}
            y2={H - pad.bottom}
            stroke="#e2e8f0"
            strokeWidth="1"
          />
          <text
            x={pad.left - 8}
            y={toY(v)}
            textAnchor="end"
            dominantBaseline="middle"
            fontSize="10"
            fill="#94a3b8"
          >
            {v.toFixed(1)}
          </text>
          <text
            x={toX(v)}
            y={H - pad.bottom + 14}
            textAnchor="middle"
            fontSize="10"
            fill="#94a3b8"
          >
            {v.toFixed(1)}
          </text>
        </g>
      ))}
      <line
        x1={toX(minV)}
        y1={toY(minV)}
        x2={toX(maxV)}
        y2={toY(maxV)}
        stroke="#94a3b8"
        strokeWidth="1.5"
        strokeDasharray="6 4"
      />
      {data.map((pt, i) => (
        <circle
          key={i}
          cx={toX(pt.actual)}
          cy={toY(pt.predicted)}
          r="5"
          fill={color}
          fillOpacity="0.75"
          stroke="white"
          strokeWidth="1"
        >
          <title>
            Actual: {pt.actual.toFixed(3)} | Predicted:{" "}
            {pt.predicted.toFixed(3)}
          </title>
        </circle>
      ))}
      <text
        x={W / 2}
        y={H - 5}
        textAnchor="middle"
        fontSize="11"
        fill="#64748b"
        fontWeight="600"
      >
        Actual {targetLabel}
      </text>
      <text
        x={14}
        y={H / 2}
        textAnchor="middle"
        fontSize="11"
        fill="#64748b"
        fontWeight="600"
        transform={`rotate(-90,14,${H / 2})`}
      >
        Predicted {targetLabel}
      </text>
    </svg>
  );
};

const ShapChart = ({ data }: { data: ShapItem[] }) => {
  if (!data || data.length === 0) {
    return (
      <div className="text-stone-400 text-sm py-8 text-center font-medium">
        No feature importance available for this model.
      </div>
    );
  }
  const maxImp = Math.max(...data.map((d) => d.importance || 0));
  const barH = 28,
    gap = 8,
    padL = 140,
    padR = 60,
    padTop = 10;
  const H = data.length * (barH + gap) + padTop + 20;
  const W = 500;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: H }}>
      {data.map((item, i) => {
        const y = padTop + i * (barH + gap);
        const barW =
          maxImp > 0
            ? ((item.importance || 0) / maxImp) * (W - padL - padR)
            : 0;
        const hue = Math.floor(220 + (i / data.length) * 60);
        const color = `hsl(${hue}, 65%, 55%)`;
        return (
          <g key={item.feature}>
            <text
              x={padL - 8}
              y={y + barH / 2}
              textAnchor="end"
              dominantBaseline="middle"
              fontSize="11"
              fill="#475569"
              fontFamily="monospace"
            >
              {item.feature.length > 18
                ? item.feature.slice(0, 18) + "…"
                : item.feature}
            </text>
            <rect
              x={padL}
              y={y}
              width={Math.max(barW, 2)}
              height={barH}
              rx="5"
              fill={color}
              fillOpacity="0.85"
            />
            <rect
              x={padL}
              y={y}
              width={Math.max(barW, 2)}
              height={barH / 2}
              rx="5"
              fill="white"
              fillOpacity="0.1"
            />
            <text
              x={padL + barW + 6}
              y={y + barH / 2}
              dominantBaseline="middle"
              fontSize="10"
              fill="#64748b"
              fontWeight="600"
            >
              {((item.importance || 0) * 100).toFixed(1)}%
            </text>
          </g>
        );
      })}
    </svg>
  );
};

const MetricCard = ({
  label,
  value,
  unit = "",
  highlight = false,
  good = true,
}: {
  label: string;
  value: number | null | undefined;
  unit?: string;
  highlight?: boolean;
  good?: boolean;
}) => (
  <div
    className={`rounded-xl p-3 ${highlight ? "bg-gradient-to-br from-amber-50 to-orange-50 border border-amber-200" : "bg-stone-50 border border-stone-200"}`}
  >
    <p className="text-xs font-medium text-stone-500 uppercase tracking-wider mb-1">
      {label}
    </p>
    <p
      className={`text-xl font-bold tabular-nums ${highlight ? "text-amber-700" : good ? "text-emerald-700" : "text-rose-600"}`}
    >
      {value !== null && value !== undefined && !isNaN(value)
        ? value.toFixed(4)
        : "—"}
      {unit && (
        <span className="text-xs ml-1 font-normal text-stone-400">{unit}</span>
      )}
    </p>
  </div>
);

// =====================================================================
// VIRIDIS COLOR HELPER
// =====================================================================
function viridisColor(val: number): string {
  // Map -1..1 → 0..1
  const t = Math.max(0, Math.min(1, (val + 1) / 2));
  // Four-stop viridis approximation
  const stops: [number, number, number][] = [
    [68, 1, 84], // 0.00 → deep purple
    [59, 82, 139], // 0.33 → blue
    [33, 145, 140], // 0.50 → teal
    [94, 201, 98], // 0.75 → green
    [253, 231, 37], // 1.00 → yellow
  ];
  const n = stops.length - 1;
  const idx = Math.min(Math.floor(t * n), n - 1);
  const f = t * n - idx;
  const [r1, g1, b1] = stops[idx];
  const [r2, g2, b2] = stops[idx + 1];
  const r = Math.round(r1 + (r2 - r1) * f);
  const g = Math.round(g1 + (g2 - g1) * f);
  const b = Math.round(b1 + (b2 - b1) * f);
  return `rgb(${r},${g},${b})`;
}

function textColorForBg(val: number): string {
  // Use white text for dark cells, dark text for bright cells
  const t = (val + 1) / 2;
  return t > 0.65 ? "#1e293b" : "#f1f5f9";
}

// =====================================================================
// CORRELATION HEATMAP COMPONENT
// =====================================================================
const CorrelationHeatmap = ({
  matrix,
}: {
  matrix: Record<string, Record<string, number | null>>;
}) => {
  const cols = Object.keys(matrix);
  if (cols.length < 2) return null;

  const CELL = 64;
  const LABEL_W = 130;
  const LABEL_H = 100;
  const LEGEND_W = 30;
  const LEGEND_MARGIN = 16;

  const gridW = cols.length * CELL;
  const gridH = cols.length * CELL;
  const totalW = LABEL_W + gridW + LEGEND_MARGIN + LEGEND_W + 30;
  const totalH = LABEL_H + gridH + 20;

  return (
    <div className="overflow-x-auto">
      <svg
        viewBox={`0 0 ${totalW} ${totalH}`}
        className="w-full"
        style={{ minWidth: Math.min(totalW, 900) }}
      >
        {/* Column labels (top, rotated) */}
        {cols.map((col, ci) => (
          <text
            key={`col-${ci}`}
            x={LABEL_W + ci * CELL + CELL / 2}
            y={LABEL_H - 6}
            textAnchor="start"
            fontSize="10"
            fill="#475569"
            fontFamily="monospace"
            transform={`rotate(-40,${LABEL_W + ci * CELL + CELL / 2},${LABEL_H - 6})`}
          >
            {col.length > 14 ? col.slice(0, 14) + "…" : col}
          </text>
        ))}

        {/* Row labels (left) */}
        {cols.map((col, ri) => (
          <text
            key={`row-${ri}`}
            x={LABEL_W - 8}
            y={LABEL_H + ri * CELL + CELL / 2}
            textAnchor="end"
            dominantBaseline="middle"
            fontSize="10"
            fill="#475569"
            fontFamily="monospace"
          >
            {col.length > 16 ? col.slice(0, 16) + "…" : col}
          </text>
        ))}

        {/* Cells */}
        {cols.map((row, ri) =>
          cols.map((col, ci) => {
            const raw = matrix[row]?.[col];
            const val = raw !== null && raw !== undefined ? raw : 0;
            const displayVal =
              raw !== null && raw !== undefined
                ? Math.abs(val) < 0.001 && val !== 0
                  ? val.toExponential(1)
                  : val.toFixed(2)
                : "—";
            const bg = viridisColor(val);
            const fg = textColorForBg(val);
            return (
              <g key={`${ri}-${ci}`}>
                <rect
                  x={LABEL_W + ci * CELL}
                  y={LABEL_H + ri * CELL}
                  width={CELL}
                  height={CELL}
                  fill={bg}
                  stroke="white"
                  strokeWidth="1"
                />
                <text
                  x={LABEL_W + ci * CELL + CELL / 2}
                  y={LABEL_H + ri * CELL + CELL / 2}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fontSize="9.5"
                  fill={fg}
                  fontFamily="monospace"
                  fontWeight="600"
                >
                  {displayVal}
                </text>
              </g>
            );
          }),
        )}

        {/* Colour legend bar */}
        {Array.from({ length: 100 }).map((_, i) => {
          const t = i / 99;
          const v = t * 2 - 1;
          return (
            <rect
              key={i}
              x={LABEL_W + gridW + LEGEND_MARGIN}
              y={LABEL_H + (1 - t) * gridH}
              width={LEGEND_W}
              height={gridH / 99 + 1}
              fill={viridisColor(v)}
            />
          );
        })}
        {/* Legend labels */}
        {[1, 0.5, 0, -0.5, -1].map((v) => {
          const y = LABEL_H + (1 - (v + 1) / 2) * gridH;
          return (
            <text
              key={v}
              x={LABEL_W + gridW + LEGEND_MARGIN + LEGEND_W + 5}
              y={y}
              dominantBaseline="middle"
              fontSize="9"
              fill="#64748b"
            >
              {v.toFixed(1)}
            </text>
          );
        })}
      </svg>
    </div>
  );
};

// =====================================================================
// STATISTICS TABLE COMPONENT
// =====================================================================
const fmt = (v: number | null | undefined, decimals = 3): string => {
  if (v === null || v === undefined || isNaN(v as number)) return "—";
  return (v as number).toFixed(decimals);
};

const StatisticsTable = ({
  statistics,
  columns,
}: {
  statistics: Record<string, ColumnStats>;
  columns: string[];
}) => {
  const statCols = columns.filter((c) => c in statistics);
  if (statCols.length === 0) return null;

  const STAT_HEADERS = [
    { key: "count", label: "Count" },
    { key: "missing", label: "Missing" },
    { key: "mean", label: "Mean" },
    { key: "std", label: "Std Dev" },
    { key: "min", label: "Min" },
    { key: "q25", label: "Q25" },
    { key: "median", label: "Median" },
    { key: "q75", label: "Q75" },
    { key: "max", label: "Max" },
  ];

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm min-w-[700px]">
        <thead>
          <tr className="bg-stone-50">
            <th className="px-4 py-3 text-left text-xs font-bold text-stone-500 uppercase tracking-wider sticky left-0 bg-stone-50 z-10 whitespace-nowrap">
              Feature
            </th>
            <th className="px-3 py-3 text-left text-xs font-bold text-amber-600 uppercase tracking-wider whitespace-nowrap">
              Unit
            </th>
            {STAT_HEADERS.map((h) => (
              <th
                key={h.key}
                className="px-3 py-3 text-right text-xs font-bold text-stone-500 uppercase tracking-wider whitespace-nowrap"
              >
                {h.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {statCols.map((col, ri) => {
            const s = statistics[col];
            const unit = getUnit(col);
            return (
              <tr
                key={col}
                className={`border-t border-stone-100 ${ri % 2 === 0 ? "" : "bg-stone-50/40"} hover:bg-amber-50/30 transition-colors`}
              >
                <td className="px-4 py-2.5 font-semibold text-stone-700 sticky left-0 bg-inherit whitespace-nowrap">
                  {col}
                </td>
                <td className="px-3 py-2.5 whitespace-nowrap">
                  {unit ? (
                    <span className="text-amber-600 bg-amber-50 border border-amber-200 text-[10px] font-bold px-1.5 py-0.5 rounded">
                      {unit}
                    </span>
                  ) : (
                    <span className="text-stone-300 text-xs">—</span>
                  )}
                </td>
                <td className="px-3 py-2.5 text-right tabular-nums text-stone-600">
                  {s.count}
                </td>
                <td className="px-3 py-2.5 text-right tabular-nums">
                  {s.missing > 0 ? (
                    <span className="text-rose-500 font-semibold">
                      {s.missing}{" "}
                      <span className="text-rose-400 text-[10px]">
                        ({fmt(s.missing_pct, 1)}%)
                      </span>
                    </span>
                  ) : (
                    <span className="text-emerald-500 text-xs font-medium">
                      None
                    </span>
                  )}
                </td>
                <td className="px-3 py-2.5 text-right tabular-nums text-stone-600 font-medium">
                  {fmt(s.mean)}
                </td>
                <td className="px-3 py-2.5 text-right tabular-nums text-stone-500">
                  {fmt(s.std)}
                </td>
                <td className="px-3 py-2.5 text-right tabular-nums text-stone-600">
                  {fmt(s.min)}
                </td>
                <td className="px-3 py-2.5 text-right tabular-nums text-stone-400">
                  {fmt(s.q25)}
                </td>
                <td className="px-3 py-2.5 text-right tabular-nums text-stone-600 font-semibold">
                  {fmt(s.median)}
                </td>
                <td className="px-3 py-2.5 text-right tabular-nums text-stone-400">
                  {fmt(s.q75)}
                </td>
                <td className="px-3 py-2.5 text-right tabular-nums text-stone-600">
                  {fmt(s.max)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

// =====================================================================
// MAIN APP
// =====================================================================
export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("upload");

  // Data
  const [fileName, setFileName] = useState<string | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [dataPreview, setDataPreview] = useState<any[]>([]);
  const [totalRows, setTotalRows] = useState<number>(0);
  const [isDragging, setIsDragging] = useState(false);
  const [dataStats, setDataStats] = useState<Record<string, ColumnStats>>({});
  const [corrMatrix, setCorrMatrix] = useState<
    Record<string, Record<string, number | null>>
  >({});

  // Training config
  const [features, setFeatures] = useState<string[]>([]);
  const [target, setTarget] = useState<string>("");
  const [testRatio, setTestRatio] = useState<number>(0.2);
  const [selectedModels, setSelectedModels] = useState<string[]>(ALL_MODELS);
  const [isTraining, setIsTraining] = useState(false);

  // Results
  const [results, setResults] = useState<TrainResponse | null>(null);
  const [selectedScatterModel, setSelectedScatterModel] = useState<string>("");
  const [selectedShapModel, setSelectedShapModel] = useState<string>("");
  const [activeMetricTab, setActiveMetricTab] = useState<"train" | "test">(
    "test",
  );

  // Prediction
  const [predictInputs, setPredictInputs] = useState<Record<string, string>>(
    {},
  );
  const [predictModel, setPredictModel] = useState<string>("");
  const [predictionResult, setPredictionResult] = useState<number | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // ------------------------------------------------------------------
  // FILE UPLOAD
  // ------------------------------------------------------------------
  const processFile = async (file: File) => {
    setFileName(file.name);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`${API_URL}/api/upload`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setColumns(data.columns);
      setDataPreview(data.preview);
      setTotalRows(data.total_rows);
      setDataStats(data.statistics ?? {});
      setCorrMatrix(data.correlation ?? {});
      setFeatures([]);
      setTarget("");
      setResults(null);
    } catch (err: any) {
      alert(
        `❌ Upload failed: ${err?.message}\n\nEnsure backend is running at http://localhost:8000`,
      );
      setColumns([]);
      setDataPreview([]);
      setTotalRows(0);
      setDataStats({});
      setCorrMatrix({});
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) processFile(file);
  }, []);

  const toggleModel = (m: string) => {
    setSelectedModels((prev) =>
      prev.includes(m)
        ? prev.length > 1
          ? prev.filter((x) => x !== m)
          : prev
        : [...prev, m],
    );
  };

  const toggleAllModels = () => {
    setSelectedModels(
      selectedModels.length === ALL_MODELS.length
        ? [ALL_MODELS[0]]
        : [...ALL_MODELS],
    );
  };

  // ------------------------------------------------------------------
  // TRAIN
  // ------------------------------------------------------------------
  const handleTrain = async () => {
    if (features.length === 0 || !target) {
      alert("Select at least one feature and a target.");
      return;
    }
    setIsTraining(true);
    try {
      const res = await fetch(`${API_URL}/api/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          features,
          target,
          test_size: testRatio,
          selected_models: selectedModels,
        }),
      });
      if (!res.ok) throw new Error(await res.text());

      const data: TrainResponse = await res.json();
      setResults(data);

      // Robust state mapping for dropdowns to handle the first available trained model
      const fallbackModel =
        data.trained_models?.length > 0 ? data.trained_models[0] : "";
      const bestOrFallback = data.best_model || fallbackModel;

      setSelectedScatterModel(bestOrFallback);
      setSelectedShapModel(bestOrFallback);
      setPredictModel(bestOrFallback);

      const resolved = data.resolved_features ?? features;
      if (data.resolved_features) setFeatures(resolved);
      if (data.resolved_target) setTarget(data.resolved_target);

      const initInputs: Record<string, string> = {};
      resolved.forEach((f: string) => (initInputs[f] = ""));
      setPredictInputs(initInputs);
    } catch (err: any) {
      let msg = "Training failed.";
      try {
        const j = JSON.parse(err?.message ?? "{}");
        msg += " " + (j?.detail ?? err?.message);
      } catch {
        msg += " " + (err?.message ?? "Check backend.");
      }
      alert(`❌ ${msg}`);
    }
    setIsTraining(false);
  };

  // ------------------------------------------------------------------
  // PREDICT
  // ------------------------------------------------------------------
  const handlePredict = async () => {
    setIsPredicting(true);
    try {
      const numInputs: Record<string, number> = {};
      for (const key in predictInputs)
        numInputs[key] = parseFloat(predictInputs[key]);

      const res = await fetch(`${API_URL}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ inputs: numInputs, model_name: predictModel }),
      });
      if (!res.ok) throw new Error(await res.text());

      const data = await res.json();
      setPredictionResult(data.prediction);
    } catch (err: any) {
      alert(`❌ Prediction failed: ${err?.message}`);
    }
    setIsPredicting(false);
  };

  // ------------------------------------------------------------------
  // PDF EXPORT  (calls backend /api/report — proper programmatic PDF)
  // ------------------------------------------------------------------
  const handleDownloadReport = async () => {
    setIsDownloading(true);
    try {
      const res = await fetch(`${API_URL}/api/report`);
      if (!res.ok) {
        const err = await res.text();
        throw new Error(err);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      // Use filename from Content-Disposition if present, else fallback
      const disposition = res.headers.get("Content-Disposition") ?? "";
      const match = disposition.match(/filename="?([^"]+)"?/);
      a.download = match ? match[1] : "RockML_Analysis_Report.pdf";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err: any) {
      alert(`❌ Report generation failed: ${err?.message ?? "Unknown error"}`);
    } finally {
      setIsDownloading(false);
    }
  };

  // ------------------------------------------------------------------
  // DERIVED STATE & SAFETY CHECKS
  // ------------------------------------------------------------------
  const trainedModels = useMemo(() => results?.trained_models ?? [], [results]);

  const r2ChartData = useMemo(() => {
    if (!results || !results.comparison) return [];
    return Object.entries(results.comparison).map(([n, m]) => ({
      name: n,
      train: m?.train?.r2,
      test: m?.test?.r2,
    }));
  }, [results]);

  const splitLabel =
    Math.round((1 - testRatio) * 100) + "/" + Math.round(testRatio * 100);

  // =====================================================================
  // RENDER
  // =====================================================================
  return (
    <div className="flex h-screen bg-[#f5f3ef] font-sans text-stone-800 overflow-hidden">
      {/* ── SIDEBAR ── */}
      <aside className="w-64 bg-stone-900 text-stone-300 flex flex-col shadow-2xl z-10 flex-shrink-0">
        <div className="px-6 py-5 border-b border-stone-800">
          <div className="flex items-center gap-2.5 mb-1">
            <Mountain className="w-7 h-7 text-amber-400" />
            <span className="text-lg font-bold text-white tracking-tight">
              RockML
            </span>
          </div>
          <p className="text-xs text-stone-500 tracking-wide">
            Rock Mechanics · AI Engine
          </p>
        </div>

        <nav className="flex-1 p-4 flex flex-col gap-1">
          {[
            {
              tab: "upload" as Tab,
              icon: <Database className="w-4 h-4" />,
              label: "Data Source",
              disabled: false,
            },
            {
              tab: "train" as Tab,
              icon: <Cpu className="w-4 h-4" />,
              label: "Train & Analyse",
              disabled: columns.length === 0,
            },
            {
              tab: "predict" as Tab,
              icon: <Zap className="w-4 h-4" />,
              label: "Predict Output",
              disabled: !results,
            },
          ].map(({ tab, icon, label, disabled }) => (
            <button
              key={tab}
              onClick={() => !disabled && setActiveTab(tab)}
              disabled={disabled}
              className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${
                activeTab === tab
                  ? "bg-amber-500 text-stone-900 shadow-md shadow-amber-900/30"
                  : disabled
                    ? "opacity-30 cursor-not-allowed text-stone-500"
                    : "hover:bg-stone-800 hover:text-white"
              }`}
            >
              {icon}
              <span>{label}</span>
              {activeTab === tab && (
                <ChevronRight className="w-3.5 h-3.5 ml-auto" />
              )}
            </button>
          ))}

          {results && (
            <div className="pt-4 mt-4 border-t border-stone-800">
              <button
                onClick={handleDownloadReport}
                disabled={isDownloading}
                className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-semibold transition-all border ${
                  isDownloading
                    ? "bg-stone-800 border-stone-700 text-stone-500 cursor-not-allowed"
                    : "bg-stone-800 text-amber-400 border-stone-700 hover:bg-stone-700 hover:text-amber-300"
                }`}
              >
                {isDownloading ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <Download className="w-4 h-4" />
                )}
                <span>Download Report</span>
              </button>
            </div>
          )}
        </nav>

        <div className="p-4 border-t border-stone-800 space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="text-stone-500">Dataset</span>
            <span className={fileName ? "text-emerald-400" : "text-stone-600"}>
              {fileName ? "Loaded ✓" : "None"}
            </span>
          </div>
          <div className="flex items-center justify-between text-xs">
            <span className="text-stone-500">Models</span>
            <span className={results ? "text-emerald-400" : "text-stone-600"}>
              {results ? `${trainedModels.length} trained` : "Not trained"}
            </span>
          </div>
          {totalRows > 0 && (
            <div className="flex items-center justify-between text-xs">
              <span className="text-stone-500">Rows</span>
              <span className="text-amber-400">
                {totalRows.toLocaleString()}
              </span>
            </div>
          )}
        </div>
      </aside>

      {/* ── MAIN ── */}
      <main className="flex-1 overflow-auto">
        <div
          id="report-container"
          className="max-w-[1400px] mx-auto p-8 space-y-8 bg-[#f5f3ef] min-h-full"
        >
          {/* TAB 1: DATA UPLOAD */}
          {activeTab === "upload" && (
            <div className="space-y-8">
              <div>
                <h2 className="text-2xl font-bold text-stone-900 mb-1">
                  Data Source
                </h2>
                <p className="text-stone-500 text-sm">
                  Upload your rock mechanics dataset (CSV or Excel)
                </p>
              </div>

              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-2xl p-12 flex flex-col items-center justify-center cursor-pointer transition-all ${
                  isDragging
                    ? "border-amber-400 bg-amber-50"
                    : "border-stone-300 hover:border-amber-400 hover:bg-amber-50/40 bg-white"
                }`}
              >
                <input
                  type="file"
                  accept=".csv,.xlsx,.xls"
                  className="hidden"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                />
                <div
                  className={`w-16 h-16 rounded-2xl flex items-center justify-center mb-4 ${isDragging ? "bg-amber-100" : "bg-stone-100"}`}
                >
                  <UploadCloud
                    className={`w-8 h-8 ${isDragging ? "text-amber-600" : "text-stone-500"}`}
                  />
                </div>
                <p className="font-semibold text-stone-700 mb-1">
                  {isDragging
                    ? "Drop your file here"
                    : "Drag & drop or click to browse"}
                </p>
                <p className="text-sm text-stone-400">
                  Supports .csv, .xlsx, .xls
                </p>
                {fileName && (
                  <div className="mt-4 px-4 py-2 bg-emerald-100 text-emerald-700 rounded-full text-sm font-medium flex items-center gap-2">
                    <CheckCircle className="w-4 h-4" /> {fileName} ·{" "}
                    {totalRows.toLocaleString()} rows loaded
                  </div>
                )}
              </div>

              {dataPreview.length > 0 && (
                <div className="bg-white rounded-2xl border border-stone-200 overflow-hidden shadow-sm">
                  <div className="px-6 py-4 border-b border-stone-100 flex items-center gap-2">
                    <Eye className="w-4 h-4 text-stone-400" />
                    <span className="font-semibold text-stone-700">
                      Data Preview (first 10 rows)
                    </span>
                    <span className="ml-auto text-xs text-stone-400">
                      {columns.length} columns
                    </span>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="bg-stone-50">
                          {columns.slice(0, 8).map((c) => (
                            <th
                              key={c}
                              className="px-4 py-3 text-left text-xs font-semibold text-stone-500 uppercase tracking-wider whitespace-nowrap"
                            >
                              {c}
                            </th>
                          ))}
                          {columns.length > 8 && (
                            <th className="px-4 py-3 text-stone-400 text-xs">
                              +{columns.length - 8} more
                            </th>
                          )}
                        </tr>
                      </thead>
                      <tbody>
                        {dataPreview.map((row, ri) => (
                          <tr
                            key={ri}
                            className="border-t border-stone-50 hover:bg-stone-50/80 transition-colors"
                          >
                            {columns.slice(0, 8).map((c) => (
                              <td
                                key={c}
                                className="px-4 py-2.5 text-stone-600 tabular-nums whitespace-nowrap"
                              >
                                {row[c] !== null && row[c] !== undefined ? (
                                  String(row[c])
                                ) : (
                                  <span className="text-stone-300">—</span>
                                )}
                              </td>
                            ))}
                            {columns.length > 8 && (
                              <td className="px-4 py-2.5 text-stone-300">…</td>
                            )}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* ── Statistical Analysis ── */}
              {Object.keys(dataStats).length > 0 && (
                <div className="bg-white rounded-2xl border border-stone-200 overflow-hidden shadow-sm">
                  <div className="px-6 py-4 border-b border-stone-100 flex items-center gap-2">
                    <BarChart2 className="w-4 h-4 text-amber-500" />
                    <span className="font-semibold text-stone-700">
                      Statistical Analysis
                    </span>
                    <span className="ml-auto text-xs text-stone-400">
                      {Object.keys(dataStats).length} numeric column
                      {Object.keys(dataStats).length !== 1 ? "s" : ""}
                    </span>
                  </div>
                  <StatisticsTable statistics={dataStats} columns={columns} />
                </div>
              )}

              {/* ── Correlation Matrix ── */}
              {Object.keys(corrMatrix).length > 1 && (
                <div className="bg-white rounded-2xl border border-stone-200 overflow-hidden shadow-sm">
                  <div className="px-6 py-4 border-b border-stone-100 flex items-center gap-2">
                    <Layers className="w-4 h-4 text-amber-500" />
                    <span className="font-semibold text-stone-700">
                      Correlation Matrix
                    </span>
                    <span className="ml-auto text-xs text-stone-400">
                      Pearson correlation between attributes
                    </span>
                  </div>
                  <div className="p-4">
                    <CorrelationHeatmap matrix={corrMatrix} />
                    <div className="flex flex-wrap items-center gap-5 mt-4 text-xs text-stone-400 justify-center">
                      <span className="flex items-center gap-1.5">
                        <span
                          className="inline-block w-3 h-3 rounded"
                          style={{ background: viridisColor(1) }}
                        />
                        Strong positive (1.0)
                      </span>
                      <span className="flex items-center gap-1.5">
                        <span
                          className="inline-block w-3 h-3 rounded"
                          style={{ background: viridisColor(0) }}
                        />
                        No correlation (0)
                      </span>
                      <span className="flex items-center gap-1.5">
                        <span
                          className="inline-block w-3 h-3 rounded"
                          style={{ background: viridisColor(-1) }}
                        />
                        Strong negative (−1.0)
                      </span>
                    </div>
                  </div>
                </div>
              )}

              <div>
                <div className="flex items-center gap-2 mb-4">
                  <BookOpen className="w-4 h-4 text-stone-400" />
                  <h3 className="font-semibold text-stone-700">
                    Rock Mechanics Test Reference
                  </h3>
                </div>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                  {DIAGRAMS.map((d) => (
                    <div
                      key={d.abbr}
                      className="bg-white rounded-xl border border-stone-200 p-4 hover:border-amber-300 hover:shadow-md transition-all group"
                    >
                      <div className="h-32 flex items-center justify-center mb-3 bg-stone-50 rounded-lg">
                        {d.component}
                      </div>
                      <div className="text-center">
                        <span className="text-xs font-bold text-amber-600 bg-amber-50 px-2 py-0.5 rounded-full">
                          {d.abbr}
                        </span>
                        <p className="text-xs text-stone-600 mt-1.5 font-medium leading-tight">
                          {d.label}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                  {ROCK_IMAGES.map((img) => (
                    <div
                      key={img.label}
                      className="relative rounded-xl overflow-hidden group h-40 bg-stone-200"
                    >
                      <img
                        src={img.url}
                        alt={img.label}
                        className="w-full h-full object-cover transition-transform group-hover:scale-105 duration-500"
                        onError={(e) => {
                          (e.target as HTMLImageElement).style.display = "none";
                        }}
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-stone-900/80 to-transparent" />
                      <div className="absolute bottom-3 left-3 flex items-center gap-1.5 text-white">
                        <span className="text-amber-400">{img.icon}</span>
                        <span className="text-xs font-semibold">
                          {img.label}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: TRAIN & ANALYSE */}
          {activeTab === "train" && (
            <div className="grid grid-cols-1 xl:grid-cols-[300px_1fr] gap-8">
              {/* LEFT CONFIG PANEL */}
              <div className="space-y-5">
                <h2 className="text-2xl font-bold text-stone-900">
                  Pipeline Setup
                </h2>

                <div className="bg-white rounded-2xl border border-stone-200 p-5 shadow-sm">
                  <label className="block text-xs font-bold text-stone-500 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                    <Target className="w-3.5 h-3.5" /> Target Variable (Y)
                  </label>
                  <select
                    value={target}
                    onChange={(e) => setTarget(e.target.value)}
                    className="w-full bg-stone-50 border border-stone-200 rounded-lg px-3 py-2.5 text-sm outline-none focus:ring-2 focus:ring-amber-400 focus:border-transparent"
                  >
                    <option value="" disabled>
                      Select target column…
                    </option>
                    {columns.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="bg-white rounded-2xl border border-stone-200 p-5 shadow-sm">
                  <label className="block text-xs font-bold text-stone-500 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                    <Layers className="w-3.5 h-3.5" /> Feature Variables (X)
                    <span className="ml-auto text-stone-400 normal-case font-normal">
                      {features.length} selected
                    </span>
                  </label>
                  <div className="space-y-1 max-h-52 overflow-y-auto pr-1">
                    {columns.map((col) => (
                      <label
                        key={col}
                        className={`flex items-center gap-2.5 px-3 py-2 rounded-lg cursor-pointer text-sm transition-colors ${features.includes(col) ? "bg-amber-50 text-amber-800" : "hover:bg-stone-50 text-stone-700"} ${col === target ? "opacity-30 pointer-events-none" : ""}`}
                      >
                        <input
                          type="checkbox"
                          checked={features.includes(col)}
                          onChange={(e) =>
                            e.target.checked
                              ? setFeatures([...features, col])
                              : setFeatures(features.filter((f) => f !== col))
                          }
                          className="w-3.5 h-3.5 rounded accent-amber-500"
                          disabled={col === target}
                        />
                        <span className="truncate">{col}</span>
                      </label>
                    ))}
                  </div>
                </div>

                <div className="bg-white rounded-2xl border border-stone-200 p-5 shadow-sm">
                  <label className="block text-xs font-bold text-stone-500 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                    <Sliders className="w-3.5 h-3.5" /> Train / Test Split
                    <span className="ml-auto text-amber-600 font-bold">
                      {splitLabel}
                    </span>
                  </label>
                  <div className="flex gap-2 flex-wrap">
                    {[
                      ["70/30", 0.3],
                      ["75/25", 0.25],
                      ["80/20", 0.2],
                      ["90/10", 0.1],
                    ].map(([lbl, val]) => (
                      <button
                        key={String(lbl)}
                        onClick={() => setTestRatio(val as number)}
                        className={`flex-1 py-2 rounded-lg text-xs font-semibold border transition-all ${testRatio === val ? "bg-amber-500 text-stone-900 border-amber-500 shadow-sm" : "border-stone-200 text-stone-600 hover:border-amber-300"}`}
                      >
                        {lbl}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="bg-white rounded-2xl border border-stone-200 p-5 shadow-sm">
                  <div className="flex items-center justify-between mb-3">
                    <label className="text-xs font-bold text-stone-500 uppercase tracking-wider flex items-center gap-1.5">
                      <Cpu className="w-3.5 h-3.5" /> Select Models
                    </label>
                    <button
                      onClick={toggleAllModels}
                      className="text-xs font-semibold text-amber-600 hover:text-amber-700 underline underline-offset-2"
                    >
                      {selectedModels.length === ALL_MODELS.length
                        ? "Deselect All"
                        : "Select All"}
                    </button>
                  </div>
                  <div className="space-y-2">
                    {ALL_MODELS.map((m) => (
                      <label
                        key={m}
                        className={`flex items-center gap-2.5 px-3 py-2.5 rounded-lg cursor-pointer text-sm border transition-all ${selectedModels.includes(m) ? "bg-stone-900 text-white border-stone-900" : "border-stone-200 hover:border-stone-300 text-stone-700"}`}
                      >
                        <input
                          type="checkbox"
                          checked={selectedModels.includes(m)}
                          onChange={() => toggleModel(m)}
                          className="w-3.5 h-3.5 rounded accent-amber-500"
                        />
                        <span className="font-medium">{m}</span>
                      </label>
                    ))}
                  </div>
                </div>

                <button
                  onClick={handleTrain}
                  disabled={isTraining || features.length === 0 || !target}
                  className={`w-full py-3.5 rounded-xl font-bold text-sm flex items-center justify-center gap-2 transition-all shadow-sm ${isTraining || features.length === 0 || !target ? "bg-stone-200 text-stone-400 cursor-not-allowed" : "bg-stone-900 hover:bg-stone-800 text-white shadow-stone-900/20"}`}
                >
                  {isTraining ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" /> Training
                      Models…
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 fill-current" /> Run Training
                      Pipeline
                    </>
                  )}
                </button>
              </div>

              {/* RIGHT RESULTS PANEL */}
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-stone-900">
                  Model Analysis
                </h2>

                {!results ? (
                  <div className="bg-white rounded-2xl border-2 border-dashed border-stone-200 h-80 flex flex-col items-center justify-center text-stone-400">
                    <BarChart2 className="w-12 h-12 mb-3 opacity-40" />
                    <p className="font-medium">
                      Configure pipeline and run training
                    </p>
                    <p className="text-sm mt-1 opacity-70">
                      Results will appear here
                    </p>
                  </div>
                ) : (
                  <div className="space-y-6">
                    {/* Training warnings (Prominent Display for XGBoost and others) */}
                    {results.training_warnings &&
                      Object.keys(results.training_warnings).length > 0 && (
                        <div className="bg-rose-50 border border-rose-200 rounded-xl p-5 shadow-sm">
                          <div className="flex items-center gap-2 text-rose-800 font-bold mb-3">
                            <AlertCircle className="w-5 h-5" /> Model Training
                            Warnings
                          </div>
                          <div className="space-y-2">
                            {Object.entries(results.training_warnings).map(
                              ([m, e]) => (
                                <p
                                  key={m}
                                  className="text-sm text-rose-700 bg-white bg-opacity-60 px-3 py-2 rounded border border-rose-100"
                                >
                                  <span className="font-bold uppercase tracking-wide mr-2">
                                    {m}:
                                  </span>{" "}
                                  {e}
                                </p>
                              ),
                            )}
                          </div>
                          <p className="text-xs text-rose-500 mt-3 font-medium">
                            Models listed above failed to train and have been
                            removed from the comparison results.
                          </p>
                        </div>
                      )}

                    <div className="bg-gradient-to-r from-stone-900 to-stone-800 rounded-2xl p-5 text-white flex items-start gap-4 shadow-md">
                      <div className="w-10 h-10 rounded-xl bg-amber-500 flex items-center justify-center flex-shrink-0">
                        <Trophy className="w-5 h-5 text-stone-900" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-xs text-stone-400 uppercase tracking-wider">
                            Best Performer
                          </span>
                          <span className="bg-amber-500 text-stone-900 text-xs font-bold px-2 py-0.5 rounded-full">
                            {results.best_model}
                          </span>
                        </div>
                        <p className="text-stone-300 text-sm leading-relaxed">
                          {results.reasoning}
                        </p>
                        <div className="flex gap-4 mt-3 text-xs text-stone-400">
                          <span>
                            Train: {results.split_info?.train_size} samples
                          </span>
                          <span>·</span>
                          <span>
                            Test: {results.split_info?.test_size} samples
                          </span>
                          <span>·</span>
                          <span>
                            Split:{" "}
                            {Math.round(
                              (1 - (results.split_info?.test_ratio ?? 0.2)) *
                                100,
                            )}
                            /
                            {Math.round(
                              (results.split_info?.test_ratio ?? 0.2) * 100,
                            )}
                          </span>
                        </div>
                      </div>
                    </div>

                    {results.pysr_equation && (
                      <div className="bg-stone-900 rounded-2xl p-5 border border-stone-700 flex items-start gap-3">
                        <FunctionSquare className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
                        <div className="min-w-0 w-full">
                          <p className="text-xs text-amber-400 font-semibold uppercase tracking-wider mb-1">
                            Symbolic Regression (PySR)
                          </p>
                          <code className="text-emerald-300 font-mono text-sm bg-stone-950 px-3 py-2 rounded-lg border border-stone-800 block break-all">
                            {results.resolved_target ?? "y"} ={" "}
                            {results.pysr_equation}
                          </code>
                        </div>
                      </div>
                    )}

                    {/* Hyperparameters per trained model */}
                    {results.hyperparameters &&
                      Object.keys(results.hyperparameters).length > 0 && (
                        <div className="bg-white rounded-2xl border border-stone-200 shadow-sm overflow-hidden">
                          <div className="px-6 py-4 border-b border-stone-100 flex items-center gap-2">
                            <Sliders className="w-4 h-4 text-amber-500" />
                            <span className="font-semibold text-stone-700">
                              Hyperparameters Used per Model
                            </span>
                          </div>
                          <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                            {Object.entries(results.hyperparameters).map(
                              ([modelName, params]) => (
                                <div
                                  key={modelName}
                                  className={`rounded-xl border p-4 ${modelName === results.best_model ? "border-amber-300 bg-amber-50/40" : "border-stone-200 bg-stone-50/40"}`}
                                >
                                  <div className="flex items-center gap-2 mb-3">
                                    <span
                                      className={`text-xs font-bold px-2 py-0.5 rounded-full ${modelName === results.best_model ? "bg-amber-500 text-stone-900" : "bg-stone-200 text-stone-700"}`}
                                    >
                                      {modelName}
                                    </span>
                                    {modelName === results.best_model && (
                                      <span className="text-xs text-amber-600 font-semibold">
                                        ★ Best
                                      </span>
                                    )}
                                  </div>
                                  {Object.keys(params).length === 0 ? (
                                    <p className="text-xs text-stone-400 italic">
                                      No hyperparameter info available.
                                    </p>
                                  ) : (
                                    <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
                                      {Object.entries(params)
                                        .filter(
                                          ([, v]) =>
                                            v !== null && v !== undefined,
                                        )
                                        .map(([k, v]) => (
                                          <div
                                            key={k}
                                            className="flex justify-between text-xs border-b border-stone-100 py-0.5"
                                          >
                                            <span className="text-stone-500 font-medium truncate mr-2">
                                              {k}
                                            </span>
                                            <span className="text-stone-800 font-semibold tabular-nums truncate text-right">
                                              {String(v)}
                                            </span>
                                          </div>
                                        ))}
                                    </div>
                                  )}
                                </div>
                              ),
                            )}
                          </div>
                        </div>
                      )}

                    {/* Train vs Test Metrics Grid */}
                    <div className="bg-white rounded-2xl border border-stone-200 shadow-sm overflow-hidden">
                      <div className="flex border-b border-stone-100">
                        <button
                          onClick={() => setActiveMetricTab("train")}
                          className={`flex-1 py-3 text-sm font-semibold transition-colors ${activeMetricTab === "train" ? "text-amber-700 bg-amber-50 border-b-2 border-amber-500" : "text-stone-500 hover:text-stone-700"}`}
                        >
                          <TrendingUp className="w-4 h-4 inline mr-1.5" />{" "}
                          Training Metrics
                        </button>
                        <button
                          onClick={() => setActiveMetricTab("test")}
                          className={`flex-1 py-3 text-sm font-semibold transition-colors ${activeMetricTab === "test" ? "text-amber-700 bg-amber-50 border-b-2 border-amber-500" : "text-stone-500 hover:text-stone-700"}`}
                        >
                          <TrendingDown className="w-4 h-4 inline mr-1.5" />{" "}
                          Testing Metrics
                        </button>
                      </div>
                      <div className="p-5 space-y-4">
                        {trainedModels.length > 0 ? (
                          trainedModels.map((name) => {
                            const metrics = results.comparison[name];
                            if (!metrics) return null;
                            const m =
                              activeMetricTab === "train"
                                ? metrics.train
                                : metrics.test;
                            const isBest = name === results.best_model;
                            return (
                              <div
                                key={name}
                                className={`rounded-xl border p-4 ${isBest ? "border-amber-200 bg-amber-50/50" : "border-stone-100"}`}
                              >
                                <div className="flex items-center gap-2 mb-3">
                                  <span className="font-semibold text-stone-800 text-sm">
                                    {name}
                                  </span>
                                  {isBest && (
                                    <span className="text-xs bg-amber-500 text-stone-900 font-bold px-2 py-0.5 rounded-full">
                                      Best
                                    </span>
                                  )}
                                </div>
                                <div className="grid grid-cols-4 gap-2">
                                  <MetricCard
                                    label="R²"
                                    value={m?.r2}
                                    highlight={isBest}
                                    good={(m?.r2 ?? 0) > 0.8}
                                  />
                                  <MetricCard
                                    label="RMSE"
                                    value={m?.rmse}
                                    good={(m?.rmse ?? 999) < 1}
                                  />
                                  <MetricCard
                                    label="MAE"
                                    value={m?.mae}
                                    good={(m?.mae ?? 999) < 1}
                                  />
                                  <MetricCard
                                    label="MSE"
                                    value={m?.mse}
                                    good={(m?.mse ?? 999) < 1}
                                  />
                                </div>
                              </div>
                            );
                          })
                        ) : (
                          <div className="text-center text-stone-500 py-8">
                            No valid models were successfully trained. Check the
                            warnings.
                          </div>
                        )}
                      </div>
                    </div>

                    {/* SHAP Feature Importance */}
                    {trainedModels.length > 0 && (
                      <div className="bg-white rounded-2xl border border-stone-200 shadow-sm p-5">
                        <div className="flex items-center justify-between mb-4">
                          <div>
                            <h3 className="font-semibold text-stone-800">
                              SHAP Feature Importance
                            </h3>
                            <p className="text-xs text-stone-400 mt-0.5">
                              Relative contribution of each feature to
                              predictions
                            </p>
                          </div>
                          <select
                            value={selectedShapModel}
                            onChange={(e) =>
                              setSelectedShapModel(e.target.value)
                            }
                            className="bg-stone-50 border border-stone-200 text-sm rounded-lg px-3 py-1.5 outline-none font-medium"
                          >
                            {trainedModels.map((m) => (
                              <option key={`shap-${m}`} value={m}>
                                {m}
                              </option>
                            ))}
                          </select>
                        </div>
                        <div className="overflow-x-auto">
                          <ShapChart
                            data={results.shap_data[selectedShapModel] ?? []}
                          />
                        </div>
                      </div>
                    )}

                    {/* R² Comparison Bar */}
                    {trainedModels.length > 0 && (
                      <div className="bg-white rounded-2xl border border-stone-200 shadow-sm p-5">
                        <h3 className="font-semibold text-stone-800 mb-4">
                          R² Score Comparison (Test Set)
                        </h3>
                        <div className="space-y-3">
                          {r2ChartData
                            .sort((a, b) => (b.test ?? -999) - (a.test ?? -999))
                            .map((item) => {
                              const pct = Math.max(
                                0,
                                Math.min(100, (item.test ?? 0) * 100),
                              );
                              const isBest = item.name === results.best_model;
                              return (
                                <div key={`r2-${item.name}`}>
                                  <div className="flex justify-between text-sm mb-1">
                                    <span className="font-medium text-stone-700">
                                      {item.name}
                                    </span>
                                    <span
                                      className={`font-bold ${isBest ? "text-amber-600" : "text-stone-600"}`}
                                    >
                                      {item.test !== undefined &&
                                      item.test !== null
                                        ? item.test.toFixed(4)
                                        : "N/A"}
                                    </span>
                                  </div>
                                  <div className="h-2.5 bg-stone-100 rounded-full overflow-hidden">
                                    <div
                                      className={`h-full rounded-full transition-all duration-700 ${isBest ? "bg-amber-500" : "bg-stone-400"}`}
                                      style={{ width: `${pct}%` }}
                                    />
                                  </div>
                                </div>
                              );
                            })}
                        </div>
                      </div>
                    )}

                    {/* Scatter Plot */}
                    {trainedModels.length > 0 && (
                      <div className="bg-white rounded-2xl border border-stone-200 shadow-sm p-5">
                        <div className="flex items-center justify-between mb-4">
                          <div>
                            <h3 className="font-semibold text-stone-800">
                              Actual vs Predicted (Test Set)
                            </h3>
                            <p className="text-xs text-stone-400 mt-0.5">
                              Points closer to dashed line = better predictions
                            </p>
                          </div>
                          <select
                            value={selectedScatterModel}
                            onChange={(e) =>
                              setSelectedScatterModel(e.target.value)
                            }
                            className="bg-stone-50 border border-stone-200 text-sm rounded-lg px-3 py-1.5 outline-none font-medium"
                          >
                            {trainedModels.map((m) => (
                              <option key={`scatter-${m}`} value={m}>
                                {m}
                              </option>
                            ))}
                          </select>
                        </div>
                        <div className="h-72">
                          <ScatterPlot
                            data={
                              results.chart_data[selectedScatterModel] ?? []
                            }
                            color={
                              selectedScatterModel === results.best_model
                                ? "#f59e0b"
                                : "#6366f1"
                            }
                            targetLabel={(() => {
                              const t = results.resolved_target ?? "Value";
                              const u = getUnit(t);
                              return u ? `${t} (${u})` : t;
                            })()}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* TAB 3: PREDICT */}
          {activeTab === "predict" && results && (
            <div className="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-8">
              <div>
                <h2 className="text-2xl font-bold text-stone-900 mb-1">
                  Inference Engine
                </h2>
                <p className="text-stone-500 text-sm mb-6">
                  Enter feature values to generate a prediction
                </p>

                <div className="bg-white rounded-2xl border border-stone-200 shadow-sm p-6">
                  <div className="mb-6 pb-5 border-b border-stone-100">
                    <label className="block text-xs font-bold text-stone-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                      <Cpu className="w-3.5 h-3.5" /> Select Trained Model
                    </label>
                    <select
                      value={predictModel}
                      onChange={(e) => {
                        setPredictModel(e.target.value);
                        setPredictionResult(null);
                      }}
                      className="w-full bg-stone-50 border border-stone-200 rounded-xl px-4 py-3 text-sm font-medium outline-none focus:ring-2 focus:ring-amber-400"
                    >
                      {trainedModels.map((m) => (
                        <option key={`predict-${m}`} value={m}>
                          {m}{" "}
                          {m === results.best_model ? "— Recommended ★" : ""}
                          {" | Test R²: "}
                          {(results.comparison[m]?.test?.r2 ?? 0).toFixed(4)}
                        </option>
                      ))}
                      {trainedModels.length === 0 && (
                        <option value="">No models available</option>
                      )}
                    </select>
                    <p className="text-xs text-stone-400 mt-2 flex items-center gap-1">
                      <Info className="w-3 h-3" />
                      Only models that succeeded in training are available
                    </p>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    {(results.resolved_features ?? []).map((feat) => {
                      const unit = getUnit(feat);
                      return (
                        <div key={feat}>
                          <label className="block text-xs font-semibold text-stone-600 mb-1.5 flex items-center gap-1.5">
                            {feat}
                            {unit && (
                              <span className="text-amber-600 bg-amber-50 border border-amber-200 text-[10px] font-bold px-1.5 py-0.5 rounded">
                                {unit}
                              </span>
                            )}
                          </label>
                          <input
                            type="number"
                            step="any"
                            value={predictInputs[feat] || ""}
                            onChange={(e) =>
                              setPredictInputs({
                                ...predictInputs,
                                [feat]: e.target.value,
                              })
                            }
                            placeholder={
                              unit ? `Value in ${unit}` : "Enter value…"
                            }
                            className="w-full bg-stone-50 border border-stone-200 rounded-lg px-3 py-2.5 text-sm outline-none focus:ring-2 focus:ring-amber-400 focus:border-transparent tabular-nums"
                          />
                        </div>
                      );
                    })}
                  </div>

                  <button
                    onClick={handlePredict}
                    disabled={isPredicting || !predictModel}
                    className={`mt-6 w-full py-3.5 rounded-xl font-bold text-sm flex items-center justify-center gap-2 transition-all ${
                      isPredicting || !predictModel
                        ? "bg-stone-200 text-stone-400 cursor-not-allowed"
                        : "bg-amber-500 hover:bg-amber-600 text-stone-900 shadow-md shadow-amber-200"
                    }`}
                  >
                    {isPredicting ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />{" "}
                        Predicting…
                      </>
                    ) : (
                      <>
                        <Zap className="w-4 h-4" /> Generate Prediction
                      </>
                    )}
                  </button>
                </div>
              </div>

              <div className="space-y-5">
                <h3 className="text-lg font-bold text-stone-800 mt-8 lg:mt-0">
                  Prediction Result
                </h3>

                {predictionResult !== null ? (
                  <div className="bg-stone-900 rounded-2xl p-8 text-center">
                    <p className="text-stone-400 text-xs uppercase tracking-widest mb-3 font-medium">
                      Predicted {results.resolved_target ?? "Target"}
                    </p>
                    <div className="text-6xl font-black text-white tabular-nums mb-1">
                      {predictionResult.toFixed(4)}
                    </div>
                    {(() => {
                      const unit = getUnit(results.resolved_target ?? "");
                      return unit ? (
                        <div className="text-amber-400 text-lg font-bold tracking-wide">
                          {unit}
                        </div>
                      ) : null;
                    })()}
                    <div className="text-stone-500 text-sm mt-4 pt-4 border-t border-stone-800">
                      via{" "}
                      <span className="text-amber-400 font-semibold">
                        {predictModel}
                      </span>{" "}
                      · Test R² ={" "}
                      <span className="text-emerald-400 font-semibold">
                        {(
                          results.comparison[predictModel]?.test?.r2 ?? 0
                        ).toFixed(4)}
                      </span>
                    </div>
                  </div>
                ) : (
                  <div className="bg-white rounded-2xl border-2 border-dashed border-stone-200 h-48 flex flex-col items-center justify-center text-stone-400">
                    <Zap className="w-10 h-10 mb-2 opacity-30" />
                    <p className="text-sm">Fill inputs & click predict</p>
                  </div>
                )}

                <div className="bg-white rounded-2xl border border-stone-200 p-4 shadow-sm">
                  <p className="text-xs font-bold text-stone-500 uppercase tracking-wider mb-3">
                    Model Quick Stats — Test Set
                  </p>
                  {predictModel && results.comparison[predictModel] ? (
                    (() => {
                      const m = results.comparison[predictModel].test;
                      const unit = getUnit(results.resolved_target ?? "");
                      return (
                        <div className="grid grid-cols-2 gap-2">
                          <MetricCard
                            label="R²"
                            value={m?.r2}
                            highlight
                            good={(m?.r2 ?? 0) > 0.8}
                          />
                          <MetricCard
                            label="RMSE"
                            value={m?.rmse}
                            unit={unit}
                            good={(m?.rmse ?? 999) < 1}
                          />
                          <MetricCard
                            label="MAE"
                            value={m?.mae}
                            unit={unit}
                            good={(m?.mae ?? 999) < 1}
                          />
                          <MetricCard
                            label="MSE"
                            value={m?.mse}
                            good={(m?.mse ?? 999) < 1}
                          />
                        </div>
                      );
                    })()
                  ) : (
                    <div className="text-sm text-stone-400 text-center py-4">
                      No stats available for the selected model.
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
