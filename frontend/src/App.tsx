import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Activity, ArrowDown, ArrowUp, ArrowUpDown, CalendarDays, Coins, Cog, ExternalLink, Filter, Flame, History, LineChart, Lock, Play, RefreshCw, Waves, X } from "lucide-react";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8003";
const GROUP_LABEL: Record<string, string> = { A_mcap30: "A · mega-cap", B_turn35: "B · movers" };
const MAX_UNFILTERED_HIST_ROWS = 200; // cap the "All months" history view so the page doesn't balloon
type Horizon = "5d" | "1d";
type PageProps = { horizon: Horizon; setHorizon: (h: Horizon) => void };

type Pick = {
  date: string; group: string; symbol: string; atm_iv: number;
  move_mag_pct: number | null; up_move?: number | null; down_move?: number | null;
  closed_opp: boolean | null; pick_rank: number; live: boolean;
  strike?: number | null;
  ce_entry?: number | null; ce_high?: number | null; ce_low?: number | null; ce_close?: number | null; ce_mult_best?: number | null;
  pe_entry?: number | null; pe_high?: number | null; pe_low?: number | null; pe_close?: number | null; pe_mult_best?: number | null;
};
type NDRow = { date: string; group: string; symbol: string; atm_iv: number; pred_move_pct: number; next_signed_pct: number | null; next_move_pct: number | null; live: boolean; picked?: boolean; rank?: number };
type MRow = { rank: number; symbol: string; atm_iv: number; live: boolean; picked: boolean; move5?: number | null; pred_move?: number | null; actual?: number | null };
type IdxRow = { label: string; move: number | null; live: boolean; pred?: number | null };
type GroupMetric = { per_yr: number; move_ge6_pct: number; move_ge8_pct: number; closed_opp_pct: number; coverage: string; top5_share_pct: number; top5_names: string[] };
type Manifest = { version: string; groups: Record<string, number>; book_metrics_2024_26: Record<string, GroupMetric> };
type PricePoint = { date: string; open: number; close: number; high: number; low: number; volume: number; delivQty: number; picked: boolean };
type Timeframe = "D" | "W" | "M";
const TF_LABEL: Record<Timeframe, string> = { D: "Daily", W: "Weekly", M: "Monthly" };
type Status = { book?: { modified: number | null; rows: number | null }; premiums?: { modified: number | null; rows: number | null }; version?: string; selector?: Record<string, unknown>; jobs: Record<string, { status: string; module?: string; tail?: string }> };

function pct(v: unknown, d = 1): string { const n = Number(v ?? 0); return Number.isFinite(n) ? `${n.toFixed(d)}%` : "-"; }
function num(v: unknown, d = 1): string { const n = Number(v); return Number.isFinite(n) ? n.toFixed(d) : "—"; }
async function getJson<T>(p: string): Promise<T> { const r = await fetch(`${API_BASE}${p}`); if (!r.ok) throw new Error(`${r.status} ${r.statusText}`); return r.json(); }
async function postJson<T>(p: string): Promise<T> { const r = await fetch(`${API_BASE}${p}`, { method: "POST" }); if (!r.ok) throw new Error(`${r.status}`); return r.json(); }
function median(vals: number[]): number { if (!vals.length) return 0; const s = [...vals].sort((a, b) => a - b); const mid = Math.floor(s.length / 2); return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2; }
const MONTH_FMT = new Intl.DateTimeFormat("en-US", { month: "short", year: "2-digit", timeZone: "UTC" });
function monthsOf<T>(rows: T[], dateOf: (r: T) => string): string[] {
  return Array.from(new Set(rows.map((r) => dateOf(r).slice(0, 7)))).sort().reverse();
}
// Month filter for history tables — a dropdown (beside the stock filter), built from the
// dates actually present rather than listing every possible month.
function MonthFilter({ months, value, onChange }: { months: string[]; value: string; onChange: (m: string) => void }) {
  if (!months.length) return null;
  return (
    <label className="chk"><CalendarDays size={15} />
      <select value={value} onChange={(e) => onChange(e.target.value)} style={{ minWidth: 130 }}>
        <option value="">All months</option>
        {months.map((m) => <option key={m} value={m}>{MONTH_FMT.format(new Date(`${m}-01T00:00:00Z`))}</option>)}
      </select>
    </label>
  );
}

const TABS = [
  { id: "sell", label: "Sell Signals", icon: <Coins size={16} /> },
  { id: "cash", label: "Cash Signals", icon: <Waves size={16} /> },
  { id: "indices", label: "Indices", icon: <Activity size={16} /> },
  { id: "desk", label: "Buy Signals", icon: <Flame size={16} /> },
  { id: "movers", label: "Universe", icon: <Filter size={16} /> },
  { id: "price", label: "Chart", icon: <LineChart size={16} /> },
  { id: "ops", label: "Refresh / Retrain", icon: <Cog size={16} /> }
];

function urlParam(name: string): string | null {
  return new URLSearchParams(window.location.search).get(name);
}

function HorizonFilter({ horizon, setHorizon }: PageProps) {
  return (
    <label><Filter size={16} />
      <select value={horizon} onChange={(e) => setHorizon(e.target.value as Horizon)}>
        <option value="5d">5-day predictions</option>
        <option value="1d">1-day predictions</option>
      </select>
    </label>
  );
}

function UMove({ v, live }: { v: number | null | undefined; live: boolean }) {
  if (live || v == null) return <span className="hint">pending</span>;
  const up = v >= 0;
  return <span className={up ? "move-up" : "move-down"}>{up ? "+" : "−"}{Math.abs(v).toFixed(1)}%</span>;
}

export default function App() {
  const [tab, setTab] = useState(() => urlParam("tab") || "sell");
  const [horizon, setHorizon] = useState<Horizon>("5d");
  const [version, setVersion] = useState("prod");
  const initialSymbol = useMemo(() => urlParam("symbol") ?? undefined, []);
  useEffect(() => { getJson<Manifest>("/prod2/manifest").then((m) => setVersion(m.version)).catch(() => {}); }, []);
  const props = { horizon, setHorizon };
  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>Koscine 3.0 — Large-Move Book</h1>
          <p>{horizon === "1d" ? "1-day movement model" : "5-day direction-agnostic mover engine"} · {version}</p>
        </div>
        <span className="lock-chip" title="Locked production engine"><Lock size={14} /> {version}</span>
      </header>
      <nav className="tabs">
        {TABS.map((t) => (
          <button key={t.id} className={`tab ${tab === t.id ? "active" : ""}`} onClick={() => setTab(t.id)}>{t.icon} {t.label}</button>
        ))}
      </nav>
      {tab === "sell" && <SellSignalsTab />}
      {tab === "cash" && <CashSignals />}
      {tab === "indices" && <IndicesSignals />}
      {tab === "desk" && <SignalDesk {...props} />}
      {tab === "movers" && <DailyMovers {...props} />}
      {tab === "price" && <PriceHistory {...props} initialSymbol={initialSymbol} />}
      {tab === "ops" && <RunRetrain />}
    </main>
  );
}

function openChartInNewWindow(symbol: string): void {
  const url = `${window.location.origin}${window.location.pathname}?tab=price&symbol=${encodeURIComponent(symbol)}`;
  window.open(url, "_blank", "noopener");
}

// Clickable stock symbol -> opens the chart-preview modal; used across Sell/Buy Signal tables.
function SymbolLink({ symbol, onOpen }: { symbol: string; onOpen: (s: string) => void }) {
  return <button type="button" className="symbol-link" onClick={() => onOpen(symbol)}>{symbol}</button>;
}

// Generic click-to-sort utility for any row array: click a column to sort by it (defaults to
// descending on first click of a new column, since most numeric columns here are "bigger is more
// interesting"), click again to flip direction. Nulls/NaN always sort last regardless of direction.
type GenericSortDir = "asc" | "desc";
function useSortedRows<T>(rows: T[], defaultKey: keyof T, defaultDir: GenericSortDir = "desc") {
  const [sort, setSort] = useState<{ key: keyof T; direction: GenericSortDir }>({ key: defaultKey, direction: defaultDir });
  const sorted = useMemo(() => {
    const copy = [...rows];
    copy.sort((a, b) => {
      const av = a[sort.key] as unknown, bv = b[sort.key] as unknown;
      const aMissing = av == null || (typeof av === "number" && !Number.isFinite(av));
      const bMissing = bv == null || (typeof bv === "number" && !Number.isFinite(bv));
      if (aMissing && bMissing) return 0;
      if (aMissing) return 1;
      if (bMissing) return -1;
      let cmp: number;
      if (typeof av === "string" && typeof bv === "string") cmp = av.localeCompare(bv);
      else if (typeof av === "boolean" && typeof bv === "boolean") cmp = av === bv ? 0 : av ? 1 : -1;
      else cmp = (av as number) < (bv as number) ? -1 : (av as number) > (bv as number) ? 1 : 0;
      return sort.direction === "asc" ? cmp : -cmp;
    });
    return copy;
  }, [rows, sort]);
  function onSort(key: keyof T) {
    setSort((s) => (s.key === key ? { key, direction: s.direction === "asc" ? "desc" : "asc" } : { key, direction: "desc" }));
  }
  return { sorted, sort, onSort };
}
function SortTh<T>({ label, sortKey, sort, onSort, className }: {
  label: React.ReactNode; sortKey: keyof T; sort: { key: keyof T; direction: GenericSortDir }; onSort: (k: keyof T) => void; className?: string;
}) {
  const active = sort.key === sortKey;
  const Icon = active ? (sort.direction === "asc" ? ArrowUp : ArrowDown) : ArrowUpDown;
  return (
    <th className={className} aria-sort={active ? (sort.direction === "asc" ? "ascending" : "descending") : "none"}>
      <button type="button" className="sort-header" onClick={() => onSort(sortKey)} aria-label={`Sort by ${label}`}>
        {label}<Icon size={14} aria-hidden="true" />
      </button>
    </th>
  );
}

// Modal chart preview: reuses the same Candles renderer as the Chart tab, for a quick look
// without leaving the current tab. "Open in new window" deep-links to the full Chart tab.
function ChartModal({ symbol, onClose }: { symbol: string; onClose: () => void }) {
  const [series, setSeries] = useState<PricePoint[]>([]);
  useEffect(() => {
    getJson<{ series: PricePoint[] }>(`/prod2/price_history?symbol=${symbol}&days=2520`)
      .then((d) => setSeries(d.series)).catch(() => setSeries([]));
  }, [symbol]);
  const weeklyFull = useMemo(() => resample(series, "W"), [series]);
  const autoDD = useMemo(() => computeAutoDD(series), [series]);
  const minDD = autoDD != null ? Math.round(autoDD * 100 * 10) / 10 : 20;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-panel" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>{symbol} — daily candles</h2>
          <div className="modal-actions">
            <button type="button" title="Open in new window" onClick={() => openChartInNewWindow(symbol)}><ExternalLink size={16} /></button>
            <button type="button" title="Close" onClick={onClose}><X size={16} /></button>
          </div>
        </div>
        <div style={{ padding: 14 }}>
          {series.length
            ? <Candles series={series} weeklyFull={weeklyFull} showLevels minDDpct={minDD} defaultCandles={DEFAULT_CANDLES_BY_TF.D} />
            : <p className="hint" style={{ padding: 16 }}>Loading…</p>}
        </div>
      </div>
    </div>
  );
}

// ----------------------------------------------------------------- Signal desk (one primary list; target-labelled overlays)
type DeskRow = {
  group: string; symbol: string; rank: number; atm_iv: number; live: boolean; conv_pctile?: number;
  atm2_contracts?: number; expensive?: boolean; v2_pick?: boolean; pick_rank?: number | null;
  pred_move_pct?: number | null; actual_peak_pct?: number | null; actual_peak_signed_pct?: number | null;
  next_day_peak_expected_pct?: number | null; next_day_close_expected_pct?: number | null;
  next_day_peak_actual_pct?: number | null; next_day_peak_signed_actual_pct?: number | null;
  next_day_close_actual_pct?: number | null; next_day_close_signed_actual_pct?: number | null; lean?: string | null; dir_conf?: number | null;
};
type DeskResp = { date: string | null; horizon: Horizon; live?: boolean; signals: DeskRow[];
  target_notes: Record<string, string>; decision: Record<string, string> };
type DeskSortKey = "group" | "rank" | "symbol" | "conviction" | "iv" | "liquidity" | "v2" | "forecast" | "actual" | "peak" | "close" | "lean";
type SignalHistRow = {
  date: string; group: string; symbol: string; rank: number; conv_pctile: number | null; atm_iv: number;
  pred_move_pct: number; actual_move_pct: number; actual_move_signed_pct: number; hit: boolean;
};
type SignalHist = {
  rows: SignalHistRow[];
  summary: { n: number; hit_rate: number; mean_pred_pct: number; mean_actual_pct: number; median_actual_pct: number; worst_actual_pct: number } | null;
};
function DeskSortHeader({ label, sortKey, sort, onSort }: { label: string; sortKey: DeskSortKey; sort: { key: DeskSortKey; direction: SortDirection }; onSort: (key: DeskSortKey) => void }) {
  const active = sort.key === sortKey; const Icon = active ? (sort.direction === "asc" ? ArrowUp : ArrowDown) : ArrowUpDown;
  return <th aria-sort={active ? (sort.direction === "asc" ? "ascending" : "descending") : "none"}><button type="button" className="sort-header" onClick={() => onSort(sortKey)} aria-label={`Sort by ${label}`}>{label}<Icon size={14} aria-hidden="true" /></button></th>;
}
function SignedActual({ value, live }: { value: number | null | undefined; live: boolean }) {
  if (live || value == null || !Number.isFinite(value)) return <span className="hint">pending</span>;
  return <span className={value >= 0 ? "move-up" : "move-down"}>{value >= 0 ? "+" : "−"}{Math.abs(value).toFixed(2)}%</span>;
}

function SignalDesk({ horizon, setHorizon }: PageProps) {
  const [dates, setDates] = useState<string[]>([]);
  const [date, setDate] = useState("");
  const [data, setData] = useState<DeskResp | null>(null);
  const [err, setErr] = useState("");
  const [sort, setSort] = useState<{ key: DeskSortKey; direction: SortDirection }>({ key: "rank", direction: "asc" });
  const [openSymbol, setOpenSymbol] = useState<string | null>(null);
  const [symbols, setSymbols] = useState<{ symbol: string; group: string }[]>([]);
  const [filterSym, setFilterSym] = useState("");
  const [filterMonth, setFilterMonth] = useState("");
  const [hist, setHist] = useState<SignalHist | null>(null);
  useEffect(() => {
    getJson<string[]>(`/prod3/dates?horizon=${horizon}`).then((d) => { setDates(d); setDate((old) => d.includes(old) ? old : d[0] || ""); }).catch((e) => setErr(String(e)));
  }, [horizon]);
  useEffect(() => {
    if (!date) return;
    setData(null);
    getJson<DeskResp>(`/signal-desk?date=${date}&horizon=${horizon}`).then(setData).catch((e) => setErr(String(e)));
  }, [date, horizon]);
  useEffect(() => { getJson<{ symbol: string; group: string }[]>("/prod2/symbols").then(setSymbols).catch(() => {}); }, []);
  useEffect(() => {
    getJson<SignalHist>(`/prod3/signal_history?horizon=${horizon}${filterSym ? `&symbol=${filterSym}` : ""}`).then(setHist).catch(() => setHist(null));
  }, [horizon, filterSym]);
  useEffect(() => setFilterMonth(""), [filterSym, horizon]);
  const allSyms = filterSym === "";
  const histMonths = useMemo(() => monthsOf(hist?.rows ?? [], (r) => r.date), [hist]);
  const histRows = useMemo(() => {
    const all = hist?.rows ?? [];
    return filterMonth ? all.filter((r) => r.date.startsWith(filterMonth)) : all;
  }, [hist, filterMonth]);
  const hs = useMemo(() => {
    if (!histRows.length) return null;
    return {
      n: histRows.length, hit_rate: histRows.filter((r) => r.hit).length / histRows.length,
      mean_pred_pct: Math.round((histRows.reduce((s, r) => s + r.pred_move_pct, 0) / histRows.length) * 100) / 100,
      mean_actual_pct: Math.round((histRows.reduce((s, r) => s + r.actual_move_pct, 0) / histRows.length) * 100) / 100,
      median_actual_pct: Math.round(median(histRows.map((r) => r.actual_move_pct)) * 100) / 100,
      worst_actual_pct: Math.round(Math.min(...histRows.map((r) => r.actual_move_pct)) * 100) / 100,
    };
  }, [histRows]);
  const rows = [...(data?.signals ?? [])].sort((a, b) => {
    const value = (r: DeskRow): string | number | null | undefined => ({ group: GROUP_LABEL[r.group] ?? r.group, rank: r.rank, symbol: r.symbol, conviction: r.conv_pctile, iv: r.atm_iv, liquidity: r.atm2_contracts, v2: r.v2_pick ? r.pick_rank ?? 99 : 999, forecast: r.pred_move_pct, actual: r.actual_peak_signed_pct, peak: r.next_day_peak_expected_pct, close: r.next_day_close_expected_pct, lean: r.lean })[sort.key];
    const av = value(a), bv = value(b); const missing = (v: unknown) => v == null || (typeof v === "number" && !Number.isFinite(v));
    if (missing(av) || missing(bv)) return missing(av) === missing(bv) ? a.symbol.localeCompare(b.symbol) : missing(av) ? 1 : -1;
    const result = typeof av === "string" && typeof bv === "string" ? av.localeCompare(bv) : Number(av) - Number(bv);
    return (sort.direction === "asc" ? 1 : -1) * (result || a.symbol.localeCompare(b.symbol));
  });
  const toggleSort = (key: DeskSortKey) => setSort((current) => ({ key, direction: current.key === key && current.direction === "asc" ? "desc" : "asc" }));
  return <>
    <section className="controls-band">
      <HorizonFilter horizon={horizon} setHorizon={setHorizon} />
      <label><CalendarDays size={16} /><select value={date} onChange={(e) => setDate(e.target.value)}>{dates.map((d, i) => <option key={d} value={d}>{d}{i === 0 ? " (latest)" : ""}</option>)}</select></label>
      <span className="hint">One shortlist. Every displayed forecast is paired only with its matching realised outcome; live rows remain pending.</span>
    </section>
    <section className="metric-strip">
      <div className="metric"><Flame size={16} /><span>Shortlist</span><strong>{rows.length} signals</strong></div>
      <div className="metric"><Lock size={16} /><span>5d baseline</span><strong>v2 ATM-IV</strong></div>
      <div className="metric"><LineChart size={16} /><span>Forecast policy</span><strong>magnitude first</strong></div>
      <div className="metric"><History size={16} /><span>Direction</span><strong>B tilt only</strong></div>
    </section>
    <section className="panel">
      <div className="panel-title"><h2>Production signal desk {date ? `· ${date}` : ""}{data?.live && <span className="pass live">live</span>}</h2><span>{horizon === "5d" ? "5-day peak-move selection" : "1-day peak-move selection"}</span></div>
      <p className="hint" style={{ padding: "10px 16px 0" }}>{horizon === "5d" ? <>The forecast and actual are both <strong>five-day peak excursions</strong>. The v2 marker is baseline overlap; it is not a second forecast.</> : <>Peak and close forecasts are separate <strong>one-day</strong> targets. A blank lean means direction-agnostic.</>}</p>
      {err && <p className="hint" style={{ padding: 16 }}>{err}</p>}
      <div className="table-wrap"><table><thead><tr><DeskSortHeader label="Group" sortKey="group" sort={sort} onSort={toggleSort} /><DeskSortHeader label="V3 rank" sortKey="rank" sort={sort} onSort={toggleSort} /><DeskSortHeader label="Symbol" sortKey="symbol" sort={sort} onSort={toggleSort} /><DeskSortHeader label="Conviction" sortKey="conviction" sort={sort} onSort={toggleSort} /><DeskSortHeader label="ATM IV" sortKey="iv" sort={sort} onSort={toggleSort} /><DeskSortHeader label="Liquidity" sortKey="liquidity" sort={sort} onSort={toggleSort} /><DeskSortHeader label="V2 baseline" sortKey="v2" sort={sort} onSort={toggleSort} />{horizon === "5d" ? <><DeskSortHeader label="5d forecast" sortKey="forecast" sort={sort} onSort={toggleSort} /><DeskSortHeader label="5d actual" sortKey="actual" sort={sort} onSort={toggleSort} /><DeskSortHeader label="B lean" sortKey="lean" sort={sort} onSort={toggleSort} /></> : <><DeskSortHeader label="V3 1d forecast" sortKey="forecast" sort={sort} onSort={toggleSort} /><DeskSortHeader label="V3 1d actual" sortKey="actual" sort={sort} onSort={toggleSort} /><DeskSortHeader label="1d peak forecast / actual" sortKey="peak" sort={sort} onSort={toggleSort} /><DeskSortHeader label="1d close forecast / actual" sortKey="close" sort={sort} onSort={toggleSort} /></>}</tr></thead>
        <tbody>{rows.map((r) => <tr key={`${r.group}-${r.symbol}`} className={r.v2_pick ? "picked-row" : undefined}>
          <td><span className="group-tag">{GROUP_LABEL[r.group] ?? r.group}</span></td><td>#{r.rank}</td><td className="symbol"><SymbolLink symbol={r.symbol} onOpen={setOpenSymbol} /></td><td>{r.conv_pctile != null ? `${(r.conv_pctile * 100).toFixed(0)}th` : "—"}</td><td>{pct(r.atm_iv * 100, 0)}</td><td>{r.atm2_contracts ? `${Math.round(r.atm2_contracts).toLocaleString()} contracts` : "—"}</td>
          <td>{r.v2_pick ? <span className="thr-badge">pick #{r.pick_rank}</span> : <span className="hint">not v2 pick</span>}</td>
          {horizon === "5d" ? <><td><strong>{r.pred_move_pct != null ? pct(r.pred_move_pct, 2) : "—"}</strong></td><td><SignedActual value={r.actual_peak_signed_pct} live={r.live} /></td><td>{r.lean ? <span className="thr-badge">{r.lean} · {pct((r.dir_conf ?? 0) * 100, 0)}</span> : <span className="hint">agnostic</span>}</td></> : <><td><strong>{r.pred_move_pct != null ? pct(r.pred_move_pct, 2) : "—"}</strong></td><td><SignedActual value={r.actual_peak_signed_pct} live={r.live} /></td><td>{r.next_day_peak_expected_pct != null ? <>{pct(r.next_day_peak_expected_pct, 2)} / <SignedActual value={r.next_day_peak_signed_actual_pct} live={r.live} /></> : "—"}</td><td>{r.next_day_close_expected_pct != null ? <>{pct(r.next_day_close_expected_pct, 2)} / <SignedActual value={r.next_day_close_signed_actual_pct} live={r.live} /></> : "—"}</td></>}
        </tr>)}{!rows.length && <tr><td className="empty-cell" colSpan={horizon === "5d" ? 10 : 11}>Loading or no signals</td></tr>}</tbody></table></div>
    </section>
    <section className="panel cockpit" style={{ marginTop: 14 }}><div className="panel-title"><h2>Production decision policy</h2><span>guardrail</span></div><p className="hint" style={{ padding: "4px 16px 16px" }}>{data?.decision?.promotion_gate ?? "Scorecard unavailable — build the production scorecard after refresh."}</p></section>
    <section className="panel cockpit" style={{ marginTop: 14 }}>
      <div className="panel-title">
        <h2>Signal history — forecast vs realized</h2>
        <div className="panel-title-controls">
          <label className="chk"><Coins size={15} />
            <select value={filterSym} onChange={(e) => setFilterSym(e.target.value)} style={{ minWidth: 160 }}>
              <option value="">All stocks</option>
              {symbols.map((s) => <option key={s.symbol} value={s.symbol}>{s.symbol} ({GROUP_LABEL[s.group] ?? s.group})</option>)}
            </select>
          </label>
          <MonthFilter months={histMonths} value={filterMonth} onChange={setFilterMonth} />
        </div>
      </div>
      <p className="hint" style={{ padding: "10px 16px 0" }}>
        Direction-agnostic: this book forecasts move <strong>size</strong>, not side. "Hit" = realized 5-day peak |move| ≥ 6%.
      </p>
      {hs ? (
        <div className="sell-bt">{hs.n} signals · hit rate (≥6%) <b>{(hs.hit_rate * 100).toFixed(0)}%</b> ·
          mean predicted <b>{hs.mean_pred_pct}%</b> vs mean realized <b>{hs.mean_actual_pct}%</b> ·
          median realized <b>{hs.median_actual_pct}%</b> · worst realized <b>{hs.worst_actual_pct}%</b></div>
      ) : null}
      <div className="table-wrap">
        <table>
          <thead><tr>
            <th>Signal date</th>{allSyms ? <th>Symbol</th> : null}<th>Grp</th><th>V3 rank</th><th>Conviction</th><th>ATM IV</th>
            <th>Predicted move</th><th>Realized move</th><th>Hit ≥6%</th>
          </tr></thead>
          <tbody>
            {histRows.map((r, i) => (
              <tr key={`${r.symbol}-${r.date}-${i}`}>
                <td>{r.date}</td>
                {allSyms ? <td><SymbolLink symbol={r.symbol} onOpen={setOpenSymbol} /></td> : null}
                <td className="hint">{GROUP_LABEL[r.group] ?? r.group}</td>
                <td>#{r.rank}</td>
                <td>{r.conv_pctile != null ? `${(r.conv_pctile * 100).toFixed(0)}th` : "—"}</td>
                <td>{pct(r.atm_iv * 100, 0)}</td>
                <td><strong>{pct(r.pred_move_pct, 2)}</strong></td>
                <td className={r.actual_move_signed_pct >= 0 ? "move-up" : "move-down"}>
                  {r.actual_move_signed_pct >= 0 ? "+" : "−"}{Math.abs(r.actual_move_signed_pct).toFixed(2)}%
                </td>
                <td>{r.hit ? "✓" : "✕"}</td>
              </tr>
            ))}
            {!histRows.length && <tr><td colSpan={allSyms ? 9 : 8} className="empty-cell">No signals</td></tr>}
          </tbody>
        </table>
      </div>
    </section>
    {openSymbol && <ChartModal symbol={openSymbol} onClose={() => setOpenSymbol(null)} />}
  </>;
}

// ----------------------------------------------------------------- Tomorrow (expected move)
type Mover = { symbol: string; group: string; exp_move_pct: number; iv_implied_pct: number; realized20_pct: number; atm_iv: number; live: boolean };
type TomorrowResp = {
  date: string | null; rank_ic?: number | null; live?: boolean; movers: Mover[];
  context?: { date?: string; nifty_exp_move_pct?: number | null; nifty_ret_5d_pct?: number | null;
    fii_net_latest?: number | null; fii_net_5d?: number | null; fii_signal?: string } | null;
};
function signed(v: number | null | undefined, d = 2): string { const n = Number(v ?? 0); return `${n >= 0 ? "+" : "−"}${Math.abs(n).toFixed(d)}`; }
function Tomorrow() {
  const [data, setData] = useState<TomorrowResp | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => { getJson<TomorrowResp>("/prod_move/tomorrow").then(setData).catch((e) => setErr(String(e))); }, []);
  if (err) return <section className="panel"><div className="panel-title"><h2>Tomorrow</h2></div><p className="hint" style={{ padding: 16 }}>Expected-move book not built yet — run it from Refresh / Retrain. ({err})</p></section>;
  if (!data) return <section className="panel"><p className="hint" style={{ padding: 16 }}>Loading…</p></section>;
  const ctx = data.context ?? {};
  const fiiBuy = (ctx.fii_net_5d ?? 0) >= 0;
  const movers = [...(data.movers ?? [])].sort((a, b) => b.exp_move_pct - a.exp_move_pct);
  const maxMove = Math.max(...movers.map((m) => m.exp_move_pct), 1);
  return (
    <>
      <section className="controls-band">
        <CalendarDays size={16} />
        <span className="hint">Expected <strong>next-day move SIZE</strong> — the reliable signal (direction is a coin flip for these names). Use it to size positions, judge if the premium is worth it, and pick straddle/strangle vs a side.</span>
      </section>
      <section className="metric-strip">
        <div className="metric"><LineChart size={16} /><span>NIFTY expected move</span><strong>±{(ctx.nifty_exp_move_pct ?? 0).toFixed(2)}%</strong></div>
        <div className="metric"><Flame size={16} /><span>Nifty 5-day momentum</span><strong className={(ctx.nifty_ret_5d_pct ?? 0) >= 0 ? "move-up" : "move-down"}>{signed(ctx.nifty_ret_5d_pct)}%</strong></div>
        <div className="metric"><RefreshCw size={16} /><span>FII flow · 5-day (₹cr)</span><strong className={fiiBuy ? "move-up" : "move-down"}>{signed(ctx.fii_net_5d, 0)}</strong></div>
        <div className="metric"><RefreshCw size={16} /><span>FII flow · last day (₹cr)</span><strong className={(ctx.fii_net_latest ?? 0) >= 0 ? "move-up" : "move-down"}>{signed(ctx.fii_net_latest, 0)}</strong></div>
      </section>
      <section className="panel">
        <div className="panel-title">
          <h2>Tomorrow — expected move by stock {data.date ? `(from ${data.date})` : ""}{data.live && <span className="pass live"> live</span>}</h2>
          <span>{data.rank_ic != null ? `forecast skill rank-IC ${data.rank_ic.toFixed(2)}` : ""} · {movers.length} names</span>
        </div>
        <p className="hint" style={{ padding: "10px 16px 0" }}>
          Direction ≈ coin flip → trade the size (ATM straddle/strangle), or take a side only on financial/high-beta movers (see the lean in <strong>Signals (v3)</strong>). FII {fiiBuy ? "buying" : "selling"} this week.
        </p>
        <div className="table-wrap"><table>
          <thead><tr><th>Symbol</th><th>Group</th><th>Expected move (next day)</th><th>IV-implied</th><th>Realized 20d</th><th>atm IV</th></tr></thead>
          <tbody>
            {movers.map((m) => (
              <tr key={m.symbol}>
                <td className="symbol">{m.symbol}</td>
                <td className="hint">{GROUP_LABEL[m.group] ?? m.group}</td>
                <td><div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ height: 8, width: `${Math.max(8, (m.exp_move_pct / maxMove) * 110)}px`, background: "#167c80", borderRadius: 4 }} />
                  <strong>±{num(m.exp_move_pct, 2)}%</strong></div></td>
                <td>±{num(m.iv_implied_pct, 2)}%</td>
                <td>±{num(m.realized20_pct, 2)}%</td>
                <td>{(m.atm_iv * 100).toFixed(0)}%</td>
              </tr>
            ))}
            {!movers.length && <tr><td colSpan={6} className="empty-cell">No data — run the expected-move book from Refresh / Retrain</td></tr>}
          </tbody>
        </table></div>
      </section>
    </>
  );
}

// ----------------------------------------------------------------- Signals (PROD v3)
type V3Row = {
  rank: number; symbol: string; group: string; iv_group: string; expensive: boolean; atm_iv: number;
  atm2_contracts: number; c_prem: number; p_prem: number; conv_pctile: number; move_mag: number | null; live: boolean;
  lean?: string | null; p_up?: number | null; dir_conf?: number | null; dir_pctile?: number | null; exp_move_pct?: number | null; move_sign?: number | null;
};
function MoversV3() {
  const [horizon, setHorizon] = useState<Horizon>("5d");
  const [dates, setDates] = useState<string[]>([]);
  const [date, setDate] = useState("");
  const [sig, setSig] = useState<{ date: string | null; live: boolean; signals: V3Row[] }>({ date: null, live: false, signals: [] });
  type Dir = { hit_overall: number; hit_by_year: Record<string, number>; auc: number; ic: number };
  const [st, setSt] = useState<{ horizons?: Record<string, { hit_ge_6pct: number; pct_expensive: number }>; direction?: Dir | null } | null>(null);
  useEffect(() => {
    getJson<string[]>(`/prod3/dates?horizon=${horizon}`).then((d) => { setDates(d); setDate(d[0] || ""); }).catch(() => setDates([]));
    getJson<{ horizons?: Record<string, { hit_ge_6pct: number; pct_expensive: number }>; direction?: Dir | null }>("/prod3/status").then(setSt).catch(() => {});
  }, [horizon]);
  useEffect(() => {
    if (!date) return;
    getJson<{ date: string | null; live: boolean; signals: V3Row[] }>(`/prod3/signals?horizon=${horizon}&date=${date}`)
      .then(setSig).catch(() => setSig({ date: null, live: false, signals: [] }));
  }, [date, horizon]);
  const hs = st?.horizons?.[horizon];
  const thr = horizon === "1d" ? 0.04 : 0.06;
  return (
    <>
      <section className="controls-band">
        <label><Filter size={16} />
          <select value={horizon} onChange={(e) => setHorizon(e.target.value as Horizon)}>
            <option value="5d">5-day move</option><option value="1d">1-day move</option>
          </select></label>
        <label>Date <select value={date} onChange={(e) => setDate(e.target.value)}>{dates.map((d) => <option key={d} value={d}>{d}</option>)}</select></label>
        <span className="hint">top-3 per group (A mega-cap · B movers) · direction-agnostic · buy CALL or PUT per your view</span>
      </section>
      <section className="panel cockpit" style={{ marginBottom: 12 }}>
        <div className="panel-title"><h2>Signals — {sig.date ?? "—"} ({horizon}){sig.live && <span className="pass live"> live</span>}</h2>
          <span>{hs ? `hit≥6% ${hs.hit_ge_6pct} · ${Math.round(hs.pct_expensive * 100)}% expensive` : ""}</span></div>
        <p className="hint" style={{ padding: "0 16px 8px" }}><strong>HIGH-IV = expensive</strong> (needs a large move to justify) — take selectively. Liquidity-gated: ATM+2% ≥1000 contracts. <strong>A: direction is yours.</strong> {horizon === "5d" && <><strong>B: a small CALL/PUT lean</strong> (direction_v1 — market-timing tilt){st?.direction ? <> · 2026 dir-hit {((st.direction.hit_by_year?.["2026"] ?? 0) * 100).toFixed(0)}% (IC {st.direction.ic >= 0 ? "+" : ""}{st.direction.ic?.toFixed(2)}) — low conviction, size small</> : null}.</>}</p>
      </section>
      <V3Table title="A · mega-cap (top-3)" rows={sig.signals.filter((r) => r.group === "A_mcap30")} horizon={horizon} thr={thr} lean={false} />
      <V3Table title="B · movers (top-3)" rows={sig.signals.filter((r) => r.group === "B_turn35")} horizon={horizon} thr={thr} lean={horizon === "5d"} />
    </>
  );
}

function Lean({ r }: { r: V3Row }) {
  if (r.lean === "CALL") return <span style={{ color: "#16a34a", fontWeight: 700 }}>CALL{r.dir_conf != null ? <span className="hint" style={{ fontWeight: 400 }}> · {(r.dir_conf * 100).toFixed(0)}%</span> : null}</span>;
  if (r.lean === "PUT") return <span style={{ color: "#dc2626", fontWeight: 700 }}>PUT{r.dir_conf != null ? <span className="hint" style={{ fontWeight: 400 }}> · {(r.dir_conf * 100).toFixed(0)}%</span> : null}</span>;
  return <span className="hint">—</span>;
}
function V3Table({ title, rows, horizon, thr, lean }: { title: string; rows: V3Row[]; horizon: Horizon; thr: number; lean: boolean }) {
  return (
    <section className="panel cockpit">
      <div className="panel-title"><h2>{title}</h2><span>{rows.length} signal{rows.length === 1 ? "" : "s"}</span></div>
      <div className="table-wrap"><table>
        <thead><tr><th>#</th><th>Symbol</th>{lean && <th>Lean</th>}<th>exp move</th><th>IV tier</th><th>IV</th><th>ATM+2% call</th><th>ATM+2% put</th><th>contracts</th><th>conv</th><th>{horizon} move</th></tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.symbol} className={r.expensive ? "picked-row" : undefined}>
              <td>#{r.rank}</td><td className="symbol">{r.symbol}</td>
              {lean && <td><Lean r={r} /></td>}
              <td>{r.exp_move_pct != null ? <strong>±{num(r.exp_move_pct, 1)}%</strong> : <span className="hint">—</span>}</td>
              <td>{r.expensive ? <span className="thr-badge">HIGH-IV</span> : <span className="pass yes">LOW-IV</span>}</td>
              <td>{(r.atm_iv * 100).toFixed(0)}%</td>
              <td>₹{num(r.c_prem, 1)}</td><td>₹{num(r.p_prem, 1)}</td>
              <td>{num(r.atm2_contracts, 0)}</td>
              <td>{(r.conv_pctile * 100).toFixed(0)}%</td>
              <td>{r.live || r.move_mag == null ? <span className="hint">pending</span> : (() => {
                const up = (r.move_sign ?? 1) >= 0; const mag = r.move_mag * 100;
                return <span className={up ? "move-up" : "move-down"} style={mag >= thr * 100 ? { fontWeight: 800 } : undefined}>{up ? "+" : "−"}{mag.toFixed(1)}%</span>;
              })()}</td>
            </tr>
          ))}
          {!rows.length && <tr><td colSpan={lean ? 11 : 10} className="empty-cell">No signals</td></tr>}
        </tbody>
      </table></div>
    </section>
  );
}

// ----------------------------------------------------------------- Daily Movers
type MoverSortKey = "rank" | "symbol" | "atm_iv" | "pred_move" | "actual" | "move5";
type SortDirection = "asc" | "desc";

function SortHeader({
  label, sortKey, sort, onSort,
}: {
  label: string;
  sortKey: MoverSortKey;
  sort: { key: MoverSortKey; direction: SortDirection };
  onSort: (key: MoverSortKey) => void;
}) {
  const active = sort.key === sortKey;
  const Icon = active ? (sort.direction === "asc" ? ArrowUp : ArrowDown) : ArrowUpDown;
  return (
    <th aria-sort={active ? (sort.direction === "asc" ? "ascending" : "descending") : "none"}>
      <button type="button" className="sort-header" onClick={() => onSort(sortKey)} aria-label={`Sort by ${label}`}>
        {label}<Icon size={14} aria-hidden="true" />
      </button>
    </th>
  );
}

function MoverTable({ title, rows, indices, horizon }: { title: string; rows: MRow[]; indices?: IdxRow[]; horizon: Horizon }) {
  const one = horizon === "1d";
  const [sort, setSort] = useState<{ key: MoverSortKey; direction: SortDirection }>({ key: "rank", direction: "asc" });
  const sortedRows = [...rows].sort((a, b) => {
    const value = (row: MRow): string | number | null | undefined => row[sort.key];
    const av = value(a), bv = value(b);
    const aMissing = av == null || (typeof av === "number" && !Number.isFinite(av));
    const bMissing = bv == null || (typeof bv === "number" && !Number.isFinite(bv));
    if (aMissing || bMissing) return aMissing === bMissing ? a.symbol.localeCompare(b.symbol) : aMissing ? 1 : -1;
    const result = typeof av === "string" && typeof bv === "string" ? av.localeCompare(bv) : Number(av) - Number(bv);
    return (sort.direction === "asc" ? 1 : -1) * (result || a.symbol.localeCompare(b.symbol));
  });
  const toggleSort = (key: MoverSortKey) => setSort((current) => ({
    key,
    direction: current.key === key && current.direction === "asc" ? "desc" : "asc",
  }));
  return (
    <section className="panel cockpit">
      <div className="panel-title"><h2>{title}</h2><span>{rows.length} stocks</span></div>
      <div className="table-wrap">
        <table>
          <thead><tr>
            <SortHeader label="Rank" sortKey="rank" sort={sort} onSort={toggleSort} />
            <SortHeader label="Symbol" sortKey="symbol" sort={sort} onSort={toggleSort} />
            <SortHeader label="IV" sortKey="atm_iv" sort={sort} onSort={toggleSort} />
            {one ? <>
              <SortHeader label="Pred move" sortKey="pred_move" sort={sort} onSort={toggleSort} />
              <SortHeader label="Next-day" sortKey="actual" sort={sort} onSort={toggleSort} />
            </> : <SortHeader label="5d move" sortKey="move5" sort={sort} onSort={toggleSort} />}
          </tr></thead>
          <tbody>
            {(indices ?? []).map((ix) => (
              <tr key={ix.label} className="index-row">
                <td>—</td><td className="symbol">{ix.label}</td><td>—</td>
                {one ? <><td><strong>{ix.pred != null ? ix.pred.toFixed(2) + "%" : "—"}</strong></td><td><UMove v={ix.move} live={ix.live} /></td></> : <td><UMove v={ix.move} live={ix.live} /></td>}
              </tr>
            ))}
            {sortedRows.map((r) => (
              <tr key={r.symbol} className={r.picked ? "picked-row" : ""}>
                <td>#{r.rank}</td>
                <td className="symbol">{r.symbol}{r.picked ? <span className="thr-badge">pick</span> : null}</td>
                <td>{(r.atm_iv * 100).toFixed(0)}%</td>
                {one
                  ? <><td><strong>{r.pred_move != null ? r.pred_move.toFixed(2) + "%" : "—"}</strong></td><td><UMove v={r.actual} live={r.live} /></td></>
                  : <td><UMove v={r.move5} live={r.live} /></td>}
              </tr>
            ))}
            {!rows.length && <tr><td colSpan={one ? 5 : 4} className="empty-cell">No data</td></tr>}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function DailyMovers({ horizon, setHorizon }: PageProps) {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [dates, setDates] = useState<string[]>([]);
  const [date, setDate] = useState("");
  const [view, setView] = useState<{ A: MRow[]; B: MRow[]; indices: IdxRow[] } | null>(null);
  const [status, setStatus] = useState("loading");

  useEffect(() => {
    Promise.all([getJson<Manifest>("/prod2/manifest"), getJson<string[]>("/prod2/dates")])
      .then(([m, ds]) => { setManifest(m); setDates(ds); setDate((p) => p || ds[0] || ""); setStatus("ready"); })
      .catch((e) => setStatus(`error: ${(e as Error).message}`));
  }, []);
  useEffect(() => {
    if (!date) return;
    setView(null);
    if (horizon === "5d") {
      getJson<{ A: MRow[]; B: MRow[]; nifty: { move5: number | null; live: boolean }; banknifty: { move5: number | null; live: boolean } }>(`/prod2/universe_day?date=${date}`)
        .then((u) => setView({ A: u.A || [], B: u.B || [], indices: [
          { label: "NIFTY (net top-30)", move: u.nifty?.move5 ?? null, live: !!u.nifty?.live },
          { label: "BANKNIFTY (banks in top-30)", move: u.banknifty?.move5 ?? null, live: !!u.banknifty?.live }] })).catch(() => setView(null));
    } else {
      getJson<{ A: MRow[]; B: MRow[]; nifty: { move: number | null; live: boolean; pred?: number | null }; banknifty: { move: number | null; live: boolean; pred?: number | null } }>(`/prod2/nextday/universe_day?date=${date}`)
        .then((u) => setView({ A: u.A || [], B: u.B || [], indices: [
          { label: "NIFTY (net top-30)", move: u.nifty?.move ?? null, live: !!u.nifty?.live, pred: u.nifty?.pred ?? null },
          { label: "BANKNIFTY (banks in top-30)", move: u.banknifty?.move ?? null, live: !!u.banknifty?.live, pred: u.banknifty?.pred ?? null }] })).catch(() => setView(null));
    }
  }, [date, horizon]);

  const bm = manifest?.book_metrics_2024_26 ?? {};
  const one = horizon === "1d";
  return (
    <>
      <section className="controls-band">
        <HorizonFilter horizon={horizon} setHorizon={setHorizon} />
        <label><CalendarDays size={16} />
          <select value={date} onChange={(e) => setDate(e.target.value)}>
            {dates.length === 0 && <option value="">No dates</option>}
            {dates.map((d, i) => <option value={d} key={d}>{d}{i === 0 ? " (latest · live)" : ""}</option>)}
          </select>
        </label>
        <span className="hint">{status} · {one ? "ranked by predicted next-day move (calibrated, IC≈0.42)" : "ranked by implied vol; top-3/group = picks"}</span>
      </section>

      <section className="dm-grid">
        <MoverTable title="A · mega-cap (top 30)" rows={view?.A ?? []} indices={view?.indices} horizon={horizon} />
        <MoverTable title="B · movers (top 35)" rows={view?.B ?? []} horizon={horizon} />
      </section>

      <section className="panel cockpit" style={{ marginTop: 14 }}>
        <div className="panel-title"><h2>Book metrics</h2><span>{one ? "1-day model — direction-agnostic" : "2024–26 · top-3/group picks"}</span></div>
        <div className="table-wrap">
          {one ? (
            <p className="hint" style={{ padding: "12px 16px" }}>
              1-day model predicts next-day move <em>magnitude</em> (calibrated, cross-sectional IC ≈ 0.42, AUC ≈ 0.73 for ≥3% moves).
              Direction is a coin flip and is not shown. "Pred move" = forecast magnitude; "Next-day" = realized signed close-to-close.
            </p>
          ) : (
            <table>
              <thead><tr><th>group</th><th>/yr</th><th>≥6%</th><th>≥8%</th><th>whipsaw</th><th>cover</th><th>top-5 most-picked</th></tr></thead>
              <tbody>
                {Object.keys(manifest?.groups ?? {}).map((g) => {
                  const d = bm[g]; if (!d) return null;
                  return <tr key={g}><td>{GROUP_LABEL[g] ?? g}</td><td>{d.per_yr}</td>
                    <td><strong>{pct(d.move_ge6_pct)}</strong></td><td>{pct(d.move_ge8_pct)}</td>
                    <td>{pct(d.closed_opp_pct)}</td><td>{d.coverage}</td>
                    <td className="byyear">{(d.top5_names ?? []).join(", ")}</td></tr>;
                })}
              </tbody>
            </table>
          )}
        </div>
      </section>
    </>
  );
}

// --------------------------------------------------------------- Stock History
function StockHistory({ horizon, setHorizon }: PageProps) {
  const [symbols, setSymbols] = useState<{ symbol: string; group: string }[]>([]);
  const [symbol, setSymbol] = useState("ADANIENT");
  const [rows5, setRows5] = useState<Pick[]>([]);
  const [rows1, setRows1] = useState<NDRow[]>([]);
  const one = horizon === "1d";
  useEffect(() => { getJson<{ symbol: string; group: string }[]>("/prod2/symbols").then(setSymbols).catch(() => {}); }, []);
  useEffect(() => {
    if (!symbol) return;
    if (one) getJson<NDRow[]>(`/prod2/nextday/stock_history?symbol=${symbol}`).then(setRows1).catch(() => setRows1([]));
    else getJson<Pick[]>(`/prod2/stock_history?symbol=${symbol}`).then(setRows5).catch(() => setRows5([]));
  }, [symbol, one]);
  return (
    <>
      <section className="controls-band">
        <HorizonFilter horizon={horizon} setHorizon={setHorizon} />
        <label><History size={16} />
          <select value={symbol} onChange={(e) => setSymbol(e.target.value)}>
            {symbols.map((s) => <option key={s.symbol} value={s.symbol}>{s.symbol} ({GROUP_LABEL[s.group] ?? s.group})</option>)}
          </select>
        </label>
      </section>
      <section className="panel cockpit">
        <div className="panel-title"><h2>{symbol} — {one ? "1-day prediction history" : "pick history (5-day)"}</h2><span>{(one ? rows1 : rows5).length} rows</span></div>
        <div className="table-wrap">
          {one ? (
            <table>
              <thead><tr><th>Date</th><th>Group</th><th>Pick</th><th>IV</th><th>Pred move</th><th>Next-day</th><th>Result</th></tr></thead>
              <tbody>
                {rows1.map((p) => (
                  <tr key={p.date} className={p.picked ? "picked-row" : undefined}>
                    <td>{p.date}</td><td><span className="group-tag">{GROUP_LABEL[p.group] ?? p.group}</span></td>
                    <td>{p.picked ? <span className="thr-badge">pick #{p.rank}</span> : <span className="hint">—</span>}</td>
                    <td>{(p.atm_iv * 100).toFixed(0)}%</td>
                    <td><strong>{num(p.pred_move_pct)}%</strong></td>
                    <td><UMove v={p.next_signed_pct} live={p.live} /></td>
                    <td>{p.live ? <span className="pass live">live</span> : <span className={`pass ${(p.next_move_pct ?? 0) >= 2 ? "yes" : "no"}`}>{(p.next_move_pct ?? 0) >= 2 ? "≥2% move" : "small"}</span>}</td>
                  </tr>
                ))}
                {!rows1.length && <tr><td colSpan={7} className="empty-cell">No history</td></tr>}
              </tbody>
            </table>
          ) : (
            <table>
              <thead><tr><th>Date</th><th>Group</th><th>Pick</th><th>IV</th><th>5d move</th><th>Result</th><th>Strike</th><th>Call best</th><th>Put best</th></tr></thead>
              <tbody>
                {rows5.map((p) => {
                  const mv = p.move_mag_pct ?? 0;
                  return <tr key={p.date} className="picked-row">
                    <td>{p.date}</td><td><span className="group-tag">{GROUP_LABEL[p.group] ?? p.group}</span></td>
                    <td><span className="thr-badge">pick #{p.pick_rank}</span></td>
                    <td>{pct(Number(p.atm_iv) * 100, 0)}</td>
                    <td><MoveCell p={p} /></td>
                    <td>{p.live ? <span className="pass live">live</span> : <span className={`pass ${mv >= 6 ? "yes" : "no"}`}>{mv >= 6 ? "≥6%" : "small"}</span>}</td>
                    <td>{p.strike != null ? num(p.strike, 0) : "—"}</td>
                    <td>{p.ce_mult_best != null ? <span className="thr-badge">{num(p.ce_mult_best, 2)}x</span> : "—"}</td>
                    <td>{p.pe_mult_best != null ? <span className="thr-badge">{num(p.pe_mult_best, 2)}x</span> : "—"}</td>
                  </tr>;
                })}
                {!rows5.length && <tr><td colSpan={9} className="empty-cell">No history</td></tr>}
              </tbody>
            </table>
          )}
        </div>
      </section>
    </>
  );
}

// --------------------------------------------------------------- Price History
function PriceHistory({ horizon, setHorizon, initialSymbol }: PageProps & { initialSymbol?: string }) {
  const [symbols, setSymbols] = useState<{ symbol: string; group: string }[]>([]);
  const [symbol, setSymbol] = useState(initialSymbol || "ADANIENT");
  const [data, setData] = useState<{ series: PricePoint[]; premiums: Pick[] }>({ series: [], premiums: [] });
  const [nd, setNd] = useState<NDRow[]>([]);
  const [tf, setTf] = useState<Timeframe>("D");
  const [showLevels, setShowLevels] = useState(true);
  const [minDD, setMinDD] = useState(10);
  // per-stock default: average of this symbol's own >10% declines over the trailing 2 years
  // (computed once per symbol, not on pan/zoom); falls back to the fixed 10%/20% if none found.
  const autoDD = useMemo(() => computeAutoDD(data.series), [data.series]);
  useEffect(() => {
    const weeklyMonthly = autoDD != null ? autoDD * 100 : 20;
    setMinDD(Math.round((tf === "D" ? weeklyMonthly / 2 : weeklyMonthly) * 10) / 10);
  }, [tf, autoDD]);
  const one = horizon === "1d";
  const candles = useMemo(() => resample(data.series, tf), [data.series, tf]);
  // S/R is always computed on weekly bars, independent of the displayed timeframe (see Candles).
  const weeklyFull = useMemo(() => resample(data.series, "W"), [data.series]);
  useEffect(() => { getJson<{ symbol: string; group: string }[]>("/prod2/symbols").then(setSymbols).catch(() => {}); }, []);
  useEffect(() => {
    if (!symbol) return;
    getJson<{ series: PricePoint[]; premiums: Pick[] }>(`/prod2/price_history?symbol=${symbol}&days=2520`).then(setData).catch(() => setData({ series: [], premiums: [] }));
    if (one) getJson<NDRow[]>(`/prod2/nextday/stock_history?symbol=${symbol}`).then(setNd).catch(() => setNd([]));
  }, [symbol, one]);
  return (
    <>
      <section className="controls-band">
        <HorizonFilter horizon={horizon} setHorizon={setHorizon} />
        <span className="hint">latest {DEFAULT_CANDLES_BY_TF[tf] === 9999 ? "all" : DEFAULT_CANDLES_BY_TF[tf]} · ↑↓ to zoom · ← → to pan · click a bar for OHLC · double-click to reset</span>
      </section>
      <section className="panel cockpit">
        <div className="panel-title">
          <h2>{symbol} — {TF_LABEL[tf].toLowerCase()} candles</h2>
          <div className="panel-title-controls">
            <label><LineChart size={16} />
              <select value={symbol} onChange={(e) => setSymbol(e.target.value)} style={{ minWidth: 150 }}>
                {symbols.map((s) => <option key={s.symbol} value={s.symbol}>{s.symbol} ({GROUP_LABEL[s.group] ?? s.group})</option>)}
              </select>
            </label>
            <label><CalendarDays size={16} />
              <select value={tf} onChange={(e) => setTf(e.target.value as Timeframe)} style={{ minWidth: 90 }}>
                {(["D", "W", "M"] as Timeframe[]).map((t) => <option key={t} value={t}>{TF_LABEL[t]}</option>)}
              </select>
            </label>
            <label className="chk">
              <input type="checkbox" checked={showLevels} onChange={(e) => setShowLevels(e.target.checked)} /> S/R levels
            </label>
            <label title={autoDD != null ? `auto-computed from this stock's own >10% declines over the trailing 2 years (${(autoDD * 100).toFixed(1)}%)` : "no >10% decline in the trailing 2 years — using the fixed default"}>
              DD % {autoDD != null ? <span className="hint" style={{ marginLeft: 0 }}>(auto {(autoDD * 100).toFixed(0)}%)</span> : null}
              <input type="number" min={1} max={60} step={1} value={minDD} style={{ width: 54 }}
                     onChange={(e) => setMinDD(Math.max(1, Math.min(60, Number(e.target.value) || 5)))} />
            </label>
            <span>{candles.length} {tf === "D" ? "days" : tf === "W" ? "weeks" : "months"}</span>
          </div>
        </div>
        <div style={{ padding: 14 }}>
          <Candles series={candles} weeklyFull={weeklyFull} showLevels={showLevels} minDDpct={minDD} defaultCandles={DEFAULT_CANDLES_BY_TF[tf]} />
        </div>
      </section>
      <section className="panel cockpit" style={{ marginTop: 14 }}>
        <div className="panel-title"><h2>{one ? "Next-day prediction vs realized" : "ATM option premium at each pick"}</h2><span>{(one ? nd : data.premiums).length} rows</span></div>
        <div className="table-wrap">
          {one ? (
            <table>
              <thead><tr><th>Date</th><th>IV</th><th>Pred move</th><th>Next-day</th></tr></thead>
              <tbody>
                {nd.map((p) => (
                  <tr key={p.date}><td>{p.date}</td><td>{(p.atm_iv * 100).toFixed(0)}%</td>
                    <td><strong>{num(p.pred_move_pct)}%</strong></td><td><UMove v={p.next_signed_pct} live={p.live} /></td></tr>
                ))}
                {!nd.length && <tr><td colSpan={4} className="empty-cell">No data</td></tr>}
              </tbody>
            </table>
          ) : (
            <table>
              <thead><tr><th>Date</th><th>5d move</th><th>Strike</th><th>Call O/H/L/C</th><th>Put O/H/L/C</th></tr></thead>
              <tbody>
                {data.premiums.map((p) => (
                  <tr key={p.date}>
                    <td>{p.date}</td><td><MoveCell p={p} /></td>
                    <td>{p.strike != null ? num(p.strike, 0) : "—"}</td>
                    <td className="byyear"><Ohlc o={p.ce_entry} h={p.ce_high} l={p.ce_low} c={p.ce_close} m={p.ce_mult_best} /></td>
                    <td className="byyear"><Ohlc o={p.pe_entry} h={p.pe_high} l={p.pe_low} c={p.pe_close} m={p.pe_mult_best} /></td>
                  </tr>
                ))}
                {!data.premiums.length && <tr><td colSpan={5} className="empty-cell">No picks for this symbol</td></tr>}
              </tbody>
            </table>
          )}
        </div>
      </section>
    </>
  );
}

// Aggregate daily OHLC candles into weekly / monthly buckets. Input is sorted ascending by date.
function resample(series: PricePoint[], tf: Timeframe): PricePoint[] {
  if (tf === "D" || series.length === 0) return series;
  const keyOf = (d: string): string => {
    if (tf === "M") return d.slice(0, 7);            // YYYY-MM
    const dt = new Date(d + "T00:00:00Z");           // ISO week (year + week number)
    const day = (dt.getUTCDay() + 6) % 7;            // Mon=0
    dt.setUTCDate(dt.getUTCDate() - day + 3);        // Thursday of this week
    const firstThu = new Date(Date.UTC(dt.getUTCFullYear(), 0, 4));
    const week = 1 + Math.round(((dt.getTime() - firstThu.getTime()) / 86400000 - 3 + ((firstThu.getUTCDay() + 6) % 7)) / 7);
    return `${dt.getUTCFullYear()}-W${week}`;
  };
  const groups = new Map<string, PricePoint[]>();
  for (const p of series) {
    const k = keyOf(p.date);
    (groups.get(k) ?? groups.set(k, []).get(k)!).push(p);
  }
  const out: PricePoint[] = [];
  for (const arr of groups.values()) {
    out.push({
      date: arr[arr.length - 1].date,
      open: arr[0].open,
      close: arr[arr.length - 1].close,
      high: Math.max(...arr.map((p) => p.high)),
      low: Math.min(...arr.map((p) => p.low)),
      volume: arr.reduce((s, p) => s + p.volume, 0),
      delivQty: arr.reduce((s, p) => s + p.delivQty, 0),
      picked: arr.some((p) => p.picked),
    });
  }
  return out;
}

// Per-stock default drawdown threshold: the average of this stock's own sizeable peak-to-trough
// declines over the trailing 2 years of DAILY data (independent of the displayed timeframe;
// computed once per symbol, not on every pan/zoom). For each N-bar pivot high, a "decline" is
// the drop to the LOWEST low reached before price recovers back above that peak (the full
// down-leg, not just to the first minor wiggle). Tries declines > 10% first; if a calm stock has
// none, retries with > 5%; if it STILL has none (essentially flat), returns null and the caller
// falls back to the fixed 10%/20% default.
function computeAutoDD(daily: PricePoint[]): number | null {
  const N = 2;
  if (daily.length < 2 * N + 1) return null;
  const lastDate = new Date(daily[daily.length - 1].date + "T00:00:00Z");
  const cutoff = new Date(lastDate); cutoff.setUTCFullYear(cutoff.getUTCFullYear() - 2);
  const cutoffStr = cutoff.toISOString().slice(0, 10);
  const win = daily.filter((p) => p.date >= cutoffStr);
  const n = win.length;
  if (n < 2 * N + 1) return null;
  const isPivotHigh = (i: number) => { for (let j = i - N; j <= i + N; j++) if (j !== i && win[j].high >= win[i].high) return false; return true; };
  const allDeclines: number[] = [];
  for (let i = N; i < n - N; i++) {
    if (!isPivotHigh(i)) continue;
    const peak = win[i].high;
    let minLow: number | null = null;
    for (let j = i + 1; j < n; j++) {
      if (win[j].high > peak) break;          // price recovered above the peak: this leg is over
      if (minLow === null || win[j].low < minLow) minLow = win[j].low;
    }
    if (minLow !== null) allDeclines.push((peak - minLow) / peak);
  }
  for (const floor of [0.10, 0.05]) {
    const kept = allDeclines.filter((d) => d > floor);
    if (kept.length) return kept.reduce((s, x) => s + x, 0) / kept.length;
  }
  return null;
}

// Largest index i with arr[i].date <= dateStr (arr sorted ascending by date); -1 if none.
function idxOnOrBefore(arr: PricePoint[], dateStr: string): number {
  let lo = 0, hi = arr.length - 1, ans = -1;
  while (lo <= hi) { const mid = (lo + hi) >> 1; if (arr[mid].date <= dateStr) { ans = mid; lo = mid + 1; } else hi = mid - 1; }
  return ans;
}
// Smallest index i with arr[i].date >= dateStr; arr.length if none.
function idxOnOrAfter(arr: PricePoint[], dateStr: string): number {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid].date < dateStr) lo = mid + 1; else hi = mid; }
  return lo;
}

type Level = { price: number; kind: "R" | "S" | "ATH"; touches: number; firstDate: string };
type Zone = { price: number; touches: number; firstDate: string };

// Support/resistance is timeframe-independent: `vis` is always the WEEKLY-resampled history
// (see Candles), never daily/monthly-specific pivots — the same lines apply whether you're
// looking at daily, weekly or monthly candles. It's the full backward history up to (never past)
// the last visible bar's date — no look-ahead — NOT clipped to the left edge of whatever x-range
// happens to be panned/zoomed into; a separate display filter then decides whether to draw a
// given line, based on whether its price falls in the currently visible y-axis range.
// Every N-bar local pivot high/low is a raw candidate (no per-peak filter); candidates within 2%
// cluster into a price zone. Within a zone, pivots are walked in time order and only count as a
// NEW touch (vs. still being part of the same ongoing test) if, since the last accepted touch,
// price either pulled >= dd away from the zone OR stayed on the far side of it (below for
// resistance, above for support) for >= G consecutive bars — a genuine gap, not a flat re-test a
// few bars later. Only zones with >= 2 touches qualify. Returns the 2 nearest resistance zones
// above the latest close and the 2 nearest support zones below (<= 4 lines), plus the all-time-
// high zone (blue) when `ath` coincides with one of the qualifying swing highs.
function srLevels(vis: PricePoint[], dd: number, ath: number, tol = 0.02): Level[] {
  // N=2: needs to beat only its 2 nearest neighbors each side. N=3 was too wide — two genuinely
  // separate nearby peaks (e.g. 3 weeks apart on a weekly chart) could sit within each other's
  // pivot window and invalidate one another even though a real pullback separated them.
  // tol: cluster tolerance -- defaults to 2% (stocks, weekly candles). NIFTY's 5-min intraday
  // chart passes 0.001 (0.10%) instead: NIFTY moves in far tighter absolute % terms within a
  // single session than a stock does over weeks/months, so the stock-sized 2% band would
  // cluster the ENTIRE visible price range into one giant "zone".
  const N = 2, athTol = 0.02, G = 8;
  const n = vis.length;
  if (n < 2 * N + 1) return [];
  const isPivotHigh = (i: number) => { for (let j = i - N; j <= i + N; j++) if (j !== i && vis[j].high >= vis[i].high) return false; return true; };
  const isPivotLow = (i: number) => { for (let j = i - N; j <= i + N; j++) if (j !== i && vis[j].low <= vis[i].low) return false; return true; };
  const highs: { i: number; p: number }[] = [], lows: { i: number; p: number }[] = [];
  for (let i = N; i < n - N; i++) {
    if (isPivotHigh(i)) highs.push({ i, p: vis[i].high });
    if (isPivotLow(i)) lows.push({ i, p: vis[i].low });
  }
  // group raw pivots within 2% of each other into candidate price zones (member lists, not yet touch-counted)
  const clusterRaw = (pts: { i: number; p: number }[]): { i: number; p: number }[][] => {
    pts = pts.slice().sort((a, b) => a.p - b.p);
    const out: { i: number; p: number }[][] = [];
    let cl: { i: number; p: number }[] = [];
    const flush = () => { if (cl.length) out.push(cl); cl = []; };
    for (const pt of pts) { if (cl.length && Math.abs(pt.p - cl[cl.length - 1].p) / pt.p > tol) flush(); cl.push(pt); }
    flush();
    return out;
  };
  // walk a candidate zone's members in time order; count only genuinely-gapped touches (dd% pullback
  // OR >= G bars on the far side of the zone since the last accepted touch)
  const resolveZone = (members: { i: number; p: number }[], isResistance: boolean): Zone | null => {
    const byTime = members.slice().sort((a, b) => a.i - b.i);
    // The line must be a price every member actually touched, not an average none of them hit:
    // for a highs-cluster (isResistance) every wick reached AT LEAST the lowest of the group, so
    // that's the shared level; for a lows-cluster every wick dropped to AT MOST the highest of
    // the group, so that's the shared level.
    const level = isResistance ? Math.min(...byTime.map((x) => x.p)) : Math.max(...byTime.map((x) => x.p));
    let touches = 1, lastI = byTime[0].i;
    for (let k = 1; k < byTime.length; k++) {
      const curI = byTime[k].i;
      let minLow = Infinity, maxHigh = -Infinity, streak = 0, bestStreak = 0;
      for (let j = lastI + 1; j < curI; j++) {
        if (isResistance) {
          minLow = Math.min(minLow, vis[j].low);
          streak = vis[j].high < level ? streak + 1 : 0;
        } else {
          maxHigh = Math.max(maxHigh, vis[j].high);
          streak = vis[j].low > level ? streak + 1 : 0;
        }
        bestStreak = Math.max(bestStreak, streak);
      }
      const ddOk = isResistance ? minLow <= level * (1 - dd) : maxHigh >= level * (1 + dd);
      if (ddOk || bestStreak >= G) { touches++; lastI = curI; }
    }
    return touches >= 2 ? { price: level, touches, firstDate: vis[byTime[0].i].date } : null;
  };
  const resZones = clusterRaw(highs).map((m) => resolveZone(m, true)).filter((z): z is Zone => z !== null);
  const supZones = clusterRaw(lows).map((m) => resolveZone(m, false)).filter((z): z is Zone => z !== null);
  const refClose = vis[n - 1].close;   // vis is always the full history, so this is the latest close
  // Collapse same-side zones within dedupTol of each other into one, keeping whichever is nearer
  // to the current price: the LOWER price for resistance, the UPPER price for support. Scaled
  // off tol (2.5x it, same ratio stocks' 5%/2% already had) rather than a fixed 5% -- at NIFTY's
  // 5-min tol=0.001 a flat 5% would be 50x the cluster width and over-merge distinct zones.
  const dedupTol = tol * 2.5;
  const dedupBySide = (zones: Zone[], keepHighest: boolean): Zone[] => {
    const sorted = zones.slice().sort((a, b) => a.price - b.price);
    const out: Zone[] = [];
    let cl: Zone[] = [];
    const flush = () => {
      if (cl.length) out.push(keepHighest ? cl.reduce((m, z) => (z.price > m.price ? z : m)) : cl.reduce((m, z) => (z.price < m.price ? z : m)));
      cl = [];
    };
    for (const z of sorted) { if (cl.length && Math.abs(z.price - cl[cl.length - 1].price) / z.price > dedupTol) flush(); cl.push(z); }
    flush();
    return out;
  };
  // Role (R vs S) is decided by CURRENT PRICE, not by whether a zone originated from pivot highs
  // or lows: an old support shelf that price has since fallen below is the very next overhead
  // level (a textbook support/resistance flip) and must be eligible as resistance even though it
  // was built from pivot lows — so both populations are pooled before picking sides.
  const allZones = [...resZones, ...supZones];
  // 2 nearest >=2-touch zones above the close (resistance), 2 nearest below (support)
  const above = dedupBySide(allZones.filter((z) => z.touches >= 2 && z.price > refClose), false).sort((a, b) => a.price - b.price).slice(0, 2);
  const below = dedupBySide(allZones.filter((z) => z.touches >= 2 && z.price < refClose), true).sort((a, b) => b.price - a.price).slice(0, 2);
  const sel: Level[] = [];
  for (const z of above) sel.push({ price: z.price, kind: "R", touches: z.touches, firstDate: z.firstDate });
  for (const z of below) sel.push({ price: z.price, kind: "S", touches: z.touches, firstDate: z.firstDate });
  // all-time-high zone (blue) when a >=2-touch zone sits at the global ATH
  const athZone = resZones.filter((z) => z.touches >= 2 && Math.abs(z.price - ath) / ath <= athTol).sort((a, b) => b.price - a.price)[0];
  if (athZone) {
    const near = sel.find((x) => Math.abs(x.price - athZone.price) / athZone.price < 0.05);
    if (near) near.kind = "ATH";
    else sel.push({ price: athZone.price, kind: "ATH", touches: athZone.touches, firstDate: athZone.firstDate });
  }
  return sel;
}

// "Nice" rounded axis levels (steps of 1/2/5 x 10^k) between lo and hi — ~count lines.
function niceTicks(lo: number, hi: number, count = 10): number[] {
  const span = hi - lo || 1;
  const mag = Math.pow(10, Math.floor(Math.log10(span / count)));
  const norm = span / count / mag;
  // round the raw step UP to the nearest 1/2/5/10 so we land near `count` lines, not above
  const step = (norm > 5 ? 10 : norm > 2 ? 5 : norm > 1 ? 2 : 1) * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 0.001; v += step) out.push(v);
  return out;
}

// Default visible candle count per timeframe. Monthly's ~120 months of 10y history is under
// this, so "9999" effectively means "show all" via the same defWin(len - N) clamp-to-0 logic.
const DEFAULT_CANDLES_BY_TF: Record<Timeframe, number> = { D: 300, W: 150, M: 9999 };

function Candles({ series, weeklyFull, showLevels, minDDpct, defaultCandles }: {
  series: PricePoint[]; weeklyFull: PricePoint[]; showLevels: boolean; minDDpct: number; defaultCandles: number;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const defWin = (len: number) => ({ s: Math.max(0, len - defaultCandles), e: len });
  const [win, setWin] = useState<{ s: number; e: number }>(defWin(series.length));
  const [sel, setSel] = useState<number | null>(null);   // clicked bar (vis-relative index); tooltip pops on click, not hover
  // default to the most recent `defaultCandles`; reset when the series changes (symbol / timeframe)
  useEffect(() => { setWin(defWin(series.length)); setSel(null); }, [series]);

  const W = 900, H = 300, padX = 40, padTop = 14, padBot = 34;
  const total = series.length;
  const s = Math.max(0, Math.min(win.s, Math.max(0, total - 2)));
  const e = Math.max(s + 2, Math.min(win.e, total));
  const vis = series.slice(s, e);
  const n = vis.length;
  const slot = n > 0 ? (W - 2 * padX) / n : 0;
  const cw = Math.max(1, Math.min(14, slot * 0.7));
  const x = (i: number) => padX + slot * (i + 0.5);

  // S/R: full backward WEEKLY history up to (never past) the last-visible bar's date — no
  // look-ahead, but not clipped to the x-window's left edge either. Recomputes only when the
  // right edge's date changes (panning within the same right edge doesn't refire this).
  const lastVisibleDate = vis.length ? vis[vis.length - 1].date : null;
  const levels = useMemo(() => {
    if (!showLevels || !lastVisibleDate) return [];
    const endIdx = idxOnOrBefore(weeklyFull, lastVisibleDate);
    if (endIdx < 0) return [];
    const scoped = weeklyFull.slice(0, endIdx + 1);
    let a = -Infinity;
    for (const p of scoped) if (p.high > a) a = p.high;   // all-time high as of lastVisibleDate
    return srLevels(scoped, minDDpct / 100, a);
  }, [weeklyFull, lastVisibleDate, showLevels, minDDpct]);

  // native wheel listener (passive:false so we can preventDefault the page scroll)
  useEffect(() => {
    const node = wrapRef.current;
    if (!node) return;
    const onWheel = (ev: WheelEvent) => {
      ev.preventDefault();
      const rect = node.getBoundingClientRect();
      const frac = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
      const cursorIdx = s + frac * n;
      const factor = ev.deltaY < 0 ? 0.82 : 1.22;         // scroll up = zoom in
      let width = Math.round(n * factor);
      width = Math.max(6, Math.min(total, width));
      let ns = Math.round(cursorIdx - frac * width);
      ns = Math.max(0, Math.min(total - width, ns));
      setWin({ s: ns, e: ns + width });
    };
    node.addEventListener("wheel", onWheel, { passive: false });
    return () => node.removeEventListener("wheel", onWheel);
  }, [s, n, total]);

  // keyboard: left/right arrows pan, up/down arrows zoom. Attached to the chart's own wrapper
  // node (not `window`), so arrow keys only move THIS chart once it's been clicked into focus --
  // not any time an arrow key is pressed anywhere on the page. `tabIndex` on the wrapper (below)
  // is what makes it focusable.
  useEffect(() => {
    const node = wrapRef.current;
    if (!node) return;
    const onKey = (ev: KeyboardEvent) => {
      if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(ev.key)) return;
      ev.preventDefault();
      if (ev.key === "ArrowLeft" || ev.key === "ArrowRight") {
        const width = e - s;
        const step = Math.max(1, Math.round(width * 0.1));
        if (ev.key === "ArrowLeft") { const ns = Math.max(0, s - step); setWin({ s: ns, e: ns + width }); }
        else { const ne = Math.min(total, e + step); setWin({ s: ne - width, e: ne }); }
      } else {
        const factor = ev.key === "ArrowUp" ? 0.7 : 1.4;   // up = zoom in, down = zoom out
        const w = Math.max(6, Math.min(total, Math.round(n * factor)));
        const mid = s + n / 2;
        const ns = Math.max(0, Math.min(total - w, Math.round(mid - w / 2)));
        setWin({ s: ns, e: ns + w });
      }
    };
    node.addEventListener("keydown", onKey);
    return () => node.removeEventListener("keydown", onKey);
  }, [s, e, n, total]);

  if (total < 2) return <span className="hint">No data</span>;

  // y-range fits the visible candles (+4% headroom); levels outside it are hidden (not drawn)
  const rawHi = Math.max(...vis.map((p) => p.high));
  const rawLo = Math.min(...vis.map((p) => p.low));
  const pad = (rawHi - rawLo) * 0.04 || 1;
  const hi = rawHi + pad, lo = rawLo - pad;
  const y = (v: number) => padTop + (1 - (v - lo) / (hi - lo || 1)) * (H - padTop - padBot);
  const up = "#167c80", down = "#9a2431";
  const atDefault = s === Math.max(0, total - defaultCandles) && e === total;
  const grid = niceTicks(lo, hi, 10);

  const idxFromClientX = (clientX: number): number => {
    const rect = wrapRef.current!.getBoundingClientRect();
    const svgX = ((clientX - rect.left) / rect.width) * W;
    return Math.max(0, Math.min(n - 1, Math.round((svgX - padX) / slot - 0.5)));
  };

  // toolbar ‹/› buttons: same pan-by-half-window step as the keyboard ArrowLeft/ArrowRight
  // handler above, just a visible affordance for it.
  const panBy = (dir: -1 | 1) => {
    const width = e - s;
    const panStep = Math.max(1, Math.round(width * 0.5));
    if (dir < 0) { const ns = Math.max(0, s - panStep); setWin({ s: ns, e: ns + width }); }
    else { const ne = Math.min(total, e + panStep); setWin({ s: ne - width, e: ne }); }
  };

  // x-axis: ~8 evenly spaced date labels across the visible window
  const step = Math.max(1, Math.ceil(n / 8));
  const ticks: number[] = [];
  for (let i = 0; i < n; i += step) ticks.push(i);
  if (ticks[ticks.length - 1] !== n - 1) ticks.push(n - 1);

  const hp = sel != null && sel < n ? vis[sel] : null;
  // prior bar's close, looked up against the full (unwindowed) series so it's available even for
  // the leftmost visible candle, not just clamped to what happens to be on screen
  const prevClose = hp != null ? series[s + sel! - 1]?.close : null;
  const chgPct = hp != null && prevClose ? ((hp.close - prevClose) / prevClose) * 100 : null;
  const tipLeftPct = sel != null ? (x(sel) / W) * 100 : 0;
  const tipRight = tipLeftPct > 62;

  // volume + delivery pane: same x-mapping as the price chart above, its own small y-scale.
  // Each bar is drawn twice at the same x: full volume (light) then delivery on top (teal),
  // so delivery reads as a stacked segment inside the same bar, not a second bar.
  const VH = 90, vPadTop = 10, vPadBot = 18;
  const volsCr = vis.map((p) => p.volume / 1e7);
  const maxVolCr = Math.max(0.01, ...volsCr);
  const vy = (vCr: number) => vPadTop + (1 - vCr / maxVolCr) * (VH - vPadTop - vPadBot);
  const vBarBottom = VH - vPadBot;

  return (
    <div ref={wrapRef} className="candles-wrap" style={{ position: "relative" }} tabIndex={0}
         onClick={(ev) => { const idx = idxFromClientX(ev.clientX); setSel((prev) => (prev === idx ? null : idx)); }}
         onDoubleClick={() => { setWin(defWin(total)); setSel(null); }}>
      <div className="candles-toolbar">
        <button type="button" title="Pan back" disabled={s <= 0} onClick={() => panBy(-1)}>‹</button>
        <button type="button" title="Pan forward" disabled={e >= total} onClick={() => panBy(1)}>›</button>
        <button type="button" title="Reset zoom" disabled={atDefault} onClick={() => setWin(defWin(total))}>Reset</button>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="spark">
        {/* y-axis grid: light-grey lines at rounded price levels */}
        {grid.map((g, gi) => (
          <g key={`grid${gi}`}>
            <line x1={padX} x2={W - padX} y1={y(g)} y2={y(g)} stroke="#e7ecea" strokeWidth={1} />
            <text x={padX - 6} y={y(g) + 3} fontSize={10} fill="#8a978f" textAnchor="end">{num(g, 0)}</text>
          </g>
        ))}
        {hp != null ? <line x1={x(sel!)} x2={x(sel!)} y1={padTop} y2={H - padBot} stroke="#b9c6c0" strokeWidth={1} strokeDasharray="3 3" /> : null}
        {vis.map((p, i) => {
          const rising = p.close >= p.open;
          const color = rising ? up : down;
          const yo = y(p.open), yc = y(p.close);
          const top = Math.min(yo, yc), bh = Math.max(1, Math.abs(yc - yo));
          return (
            <g key={s + i}>
              <line x1={x(i)} x2={x(i)} y1={y(p.high)} y2={y(p.low)} stroke={color} strokeWidth={1} />
              <rect x={x(i) - cw / 2} y={top} width={cw} height={bh} fill={color} />
            </g>
          );
        })}
        {/* S/R zones (only those within the visible price range): first touch -> right edge.
            firstDate comes from the weekly-computed zone; resolve it to a position on whatever
            timeframe is currently displayed (series), relative to the visible window start s. */}
        {levels.filter((L) => L.price >= lo && L.price <= hi).map((L, li) => {
          const color = L.kind === "ATH" ? "#2563b0" : "#33443d";
          const relIdx = idxOnOrAfter(series, L.firstDate) - s;
          const xStart = Math.min(W - padX, Math.max(padX, x(relIdx)));
          return (
            <g key={`lv${li}`}>
              <line x1={xStart} x2={W - padX} y1={y(L.price)} y2={y(L.price)} stroke={color} strokeWidth={0.55} />
              <text x={W - padX - 4} y={y(L.price) - 3} fontSize={9} fill={color} textAnchor="end">{num(L.price, 1)} ×{L.touches}</text>
            </g>
          );
        })}
        {/* x-axis date ticks */}
        {ticks.map((i) => (
          <text key={i} x={x(i)} y={H - padBot + 16} fontSize={10} fill="#60706a"
                textAnchor={i === 0 ? "start" : i === n - 1 ? "end" : "middle"}>{vis[i].date.slice(2)}</text>
        ))}
      </svg>
      <svg viewBox={`0 0 ${W} ${VH}`} className="spark" style={{ marginTop: 4 }}>
        <text x={padX} y={9} fontSize={9} fill="#8a978f">Volume (Cr) · teal = delivery</text>
        {hp != null ? <line x1={x(sel!)} x2={x(sel!)} y1={vPadTop} y2={vBarBottom} stroke="#b9c6c0" strokeWidth={1} strokeDasharray="3 3" /> : null}
        {vis.map((p, i) => {
          const volCr = p.volume / 1e7;
          const delivCr = Math.min(p.delivQty / 1e7, volCr);
          return (
            <g key={s + i}>
              <rect x={x(i) - cw / 2} y={vy(volCr)} width={cw} height={Math.max(0, vBarBottom - vy(volCr))} fill="#cdd8d3" />
              <rect x={x(i) - cw / 2} y={vy(delivCr)} width={cw} height={Math.max(0, vBarBottom - vy(delivCr))} fill="#167c80" />
            </g>
          );
        })}
        <text x={padX - 6} y={vy(maxVolCr) + 3} fontSize={9} fill="#8a978f" textAnchor="end">{num(maxVolCr, 1)}</text>
        <text x={padX - 6} y={vBarBottom} fontSize={9} fill="#8a978f" textAnchor="end">0</text>
      </svg>
      {hp ? (
        <div className="candle-tip" style={{ [tipRight ? "right" : "left"]: `calc(${tipRight ? 100 - tipLeftPct : tipLeftPct}% + 10px)`, top: 8 }}>
          <strong>{hp.date}</strong>
          {chgPct != null ? (
            <span>Chg <b style={{ color: chgPct >= 0 ? up : down }}>{chgPct >= 0 ? "+" : ""}{num(chgPct, 2)}%</b></span>
          ) : null}
          <span>O <b>{num(hp.open, 1)}</b></span>
          <span>H <b>{num(hp.high, 1)}</b></span>
          <span>L <b>{num(hp.low, 1)}</b></span>
          <span>C <b style={{ color: hp.close >= hp.open ? up : down }}>{num(hp.close, 1)}</b></span>
          <span>Vol <b>{num(hp.volume / 1e7, 2)} Cr</b></span>
          <span>Deliv <b>{num(hp.delivQty / 1e7, 2)} Cr</b>{hp.volume > 0 ? ` (${num(hp.delivQty / hp.volume * 100, 0)}%)` : ""}</span>
        </div>
      ) : null}
    </div>
  );
}

// ----------------------------------------------------------------- Sell Strategies
type CondorRow = {
  symbol: string; group: string; expiry: string; dte: number; underlying: number; iv_ratio: number | null;
  short_ce: number; long_ce: number; short_pe: number; long_pe: number;
  oi_short_ce: number | null; oi_long_ce: number | null; oi_short_pe: number | null; oi_long_pe: number | null;
  sell_premium: number; buy_premium: number;
  credit: number; max_risk: number; lot_size: number | null; max_risk_per_lot: number | null;
  max_profit: number; max_profit_per_lot: number | null;
  ror_pct: number; be_low: number; be_high: number;
  richer_side: "CE" | "PE"; ce_credit: number; pe_credit: number; in_window: boolean;
};
type SellResp = {
  as_of: string | null; params: Record<string, number>;
  backtest: {
    window: string; ev_on_risk: number; win_rate: number; worst: string;
    best_of_day?: { window: string; n: number; ev_on_risk: number; win_rate: number; worst: string; per_week: number; median_pnl_per_lot?: number; note?: string };
  };
  candidates: CondorRow[]; top_picks: CondorRow[];
};
type SellHistRow = {
  symbol: string; group: string; signal_date: string; expiry: string; dte: number; iv_ratio: number | null;
  short_ce: number; long_ce: number; short_pe: number; long_pe: number;
  oi_short_ce: number | null; oi_long_ce: number | null; oi_short_pe: number | null; oi_long_pe: number | null;
  sell_premium: number; buy_premium: number;
  credit: number; max_risk: number; max_profit: number;
  lot_size: number | null; max_risk_per_lot: number | null; max_profit_per_lot: number | null;
  entry_ror_pct: number;   // theoretical ROR knowable at entry -- used to rank/dedup still-"pending" rows, which have no realized ror_pct yet
  exit_value: number; pnl: number | null; pnl_per_lot: number | null; ror_pct: number | null; max_dd_pct: number; outcome: string;   // outcome may be "pending" -- forward window not complete yet, see gen_sell_signal_history.py
};
type SellHist = {
  rows: SellHistRow[];
  summary: { n: number; n_pending?: number; win_rate: number; ev_ror_pct: number; median_ror_pct: number; worst_ror_pct: number; worst_dd_pct: number; total_pnl: number } | null;
};

function SellSignalsTab() {
  return <MergedSellSignals />;
}

// ----------------------------------------------------------------- Merged Sell Signals (Condor + Broken-Wing + IV Skew)
// Skew is filtered server-side to exclude any symbol already firing a Broken-Wing signal that
// day (see /prod2/skew_strategy's `excluded_by_broken_wing`) -- but Condor is NOT server-side
// de-duped against either (independent script/endpoint, no cross-exclusion), so the same symbol
// can legitimately show up from more than one strategy/tier on the same day. Per-symbol (daily)
// / per-(symbol,date) (history) dedup below collapses those down to the single most profitable
// signal, rather than showing every strategy's take on the same stock.
type MergedRow = {
  strategy: "BW" | "SIV" | "CONDOR"; tierId: string; tierLabel: string; rank: number;
  symbol: string; group: string; expiry: string; dte: number; underlying: number; iv_ratio: number | null;
  sideDesc: string; positionDesc: string; oiDesc: string;
  credit: number; max_profit: number; max_risk: number; lot_size: number | null;
  max_profit_per_lot: number | null; max_risk_per_lot: number | null; ror_pct: number;
};
type MergedHistRow = {
  strategy: "BW" | "SIV" | "CONDOR"; tierId: string; symbol: string; group: string; signal_date: string;
  expiry: string; dte: number; iv_ratio: number | null; positionDesc: string; oiDesc: string;
  credit: number; max_profit: number; max_risk: number;
  lot_size: number | null; max_risk_per_lot: number | null; max_profit_per_lot: number | null;
  entry_ror_pct: number;   // theoretical ROR knowable at entry -- used to rank/dedup still-"pending" rows, which have no realized ror_pct yet
  exit_value: number; pnl: number | null; pnl_per_lot: number | null; ror_pct: number | null; max_dd_pct: number; outcome: string;   // outcome may be "pending" -- forward window not complete yet, see gen_sell_signal_history.py
};
// OI (lots) for the strikes actually in play, formatted to match positionDesc's short/long
// pairing -- "1094/789" style (short leg / long leg). BW & Condor show both CE and PE pairs
// (mirroring positionDesc's "C x/y . P x/y"); Skew shows a single pair for whichever side sold.
function oiPair(short: number | null, long: number | null): string {
  return `${short ?? "—"}/${long ?? "—"}`;
}
const MERGED_TIER_LABEL: Record<string, string> = {
  condor: "Condor · 2%/5%",
  t1_2x6: "Tier 1 · 2%/6%", t2_3x8: "Tier 2 · 3%/8%", t3_2x10: "Tier 3 · 2%/10%",
  t1_skew35: "Skew T1 · 3.5%", t2_skew6: "Skew T2 · 6%", t3_skew8: "Skew T3 · 8%",
};
// "most profitable" for dedup, both daily AND history: absolute max profit per lot (real rupees
// the trade pays out at max profit) rather than ror_pct (% return on risk) -- ror_pct can favor a
// tiny, barely-funded spread over a much bigger absolute payout. Falls back to raw max_profit
// (points, not rupees) only when lot_size/max_profit_per_lot isn't known for that symbol. Used
// even for history rows (not realized pnl_per_lot) so a still-"pending" trade sorts the same way
// a same-day daily candidate would have -- the THEORETICAL max profit at entry, not an outcome
// that isn't known yet.
function maxProfitScore(r: { max_profit_per_lot: number | null; max_profit: number }): number {
  return r.max_profit_per_lot ?? r.max_profit;
}
// Collapse multiple signals firing on the same stock (same day) down to the single most
// profitable one -- across strategies (Condor/Broken-Wing/Skew aren't all mutually de-duped
// server-side) and across tiers within one strategy alike.
function dedupMostProfitable<T extends { symbol: string }>(rows: T[], keyOf: (r: T) => string,
                                                             score: (r: T) => number): T[] {
  const best = new Map<string, T>();
  for (const r of rows) {
    const key = keyOf(r);
    const cur = best.get(key);
    if (!cur || score(r) > score(cur)) best.set(key, r);
  }
  return Array.from(best.values());
}

function MergedSellSignals() {
  const [bwData, setBwData] = useState<BWResp | null>(null);
  const [skewData, setSkewData] = useState<SkewTieredResp | null>(null);
  const [sellData, setSellData] = useState<SellResp | null>(null);
  const [symbols, setSymbols] = useState<{ symbol: string; group: string }[]>([]);
  const [symStats, setSymStats] = useState<BWSymbolStats | null>(null);
  const [filterSym, setFilterSym] = useState("");
  const [filterStrategy, setFilterStrategy] = useState<"" | "BW" | "SIV" | "CONDOR">("");
  const [filterMonth, setFilterMonth] = useState("");
  const [bwHist, setBwHist] = useState<BWHist | null>(null);
  const [skewHist, setSkewHist] = useState<SkewTieredHist | null>(null);
  const [sellHist, setSellHist] = useState<SellHist | null>(null);
  const [openSymbol, setOpenSymbol] = useState<string | null>(null);

  useEffect(() => { getJson<BWResp>("/prod2/broken_wing_strategy").then(setBwData).catch(() => setBwData(null)); }, []);
  useEffect(() => { getJson<SkewTieredResp>("/prod2/skew_strategy").then(setSkewData).catch(() => setSkewData(null)); }, []);
  useEffect(() => { getJson<SellResp>("/prod2/sell_strategies").then(setSellData).catch(() => setSellData(null)); }, []);
  useEffect(() => { getJson<{ symbol: string; group: string }[]>("/prod2/symbols").then(setSymbols).catch(() => {}); }, []);
  useEffect(() => { getJson<BWSymbolStats>("/prod2/broken_wing_symbol_stats").then(setSymStats).catch(() => setSymStats(null)); }, []);
  useEffect(() => {
    const q = filterSym ? `?symbol=${filterSym}` : "";
    getJson<BWHist>(`/prod2/broken_wing_signal_history${q}`).then(setBwHist).catch(() => setBwHist(null));
    getJson<SkewTieredHist>(`/prod2/skew_signal_history${q}`).then(setSkewHist).catch(() => setSkewHist(null));
    getJson<SellHist>(`/prod2/sell_signal_history${q}`).then(setSellHist).catch(() => setSellHist(null));
  }, [filterSym]);
  useEffect(() => setFilterMonth(""), [filterSym, filterStrategy]);

  const bwTiers = bwData?.tiers ?? [];
  const skewTiers = skewData?.tiers ?? [];

  const dailyFlat: MergedRow[] = useMemo(() => {
    const bwRows: MergedRow[] = bwTiers.flatMap((t) => t.top_picks.map((r, i) => ({
      strategy: "BW" as const, tierId: t.id, tierLabel: t.label, rank: i + 1,
      symbol: r.symbol, group: r.group, expiry: r.expiry, dte: r.dte, underlying: r.underlying, iv_ratio: r.iv_ratio,
      sideDesc: `${r.richer_side === "CE" ? "Call" : "Put"} richer`,
      positionDesc: `C ${num(r.short_ce, 0)}/${num(r.long_ce, 0)} · P ${num(r.short_pe, 0)}/${num(r.long_pe, 0)}`,
      oiDesc: `C ${oiPair(r.oi_short_ce, r.oi_long_ce)} · P ${oiPair(r.oi_short_pe, r.oi_long_pe)}`,
      credit: r.credit, max_profit: r.max_profit, max_risk: r.max_risk, lot_size: r.lot_size,
      max_profit_per_lot: r.max_profit_per_lot, max_risk_per_lot: r.max_risk_per_lot, ror_pct: r.ror_pct,
    })));
    const skewRows: MergedRow[] = skewTiers.flatMap((t) => t.top_picks.map((r, i) => ({
      strategy: "SIV" as const, tierId: t.id, tierLabel: t.label, rank: i + 1,
      symbol: r.symbol, group: r.group, expiry: r.expiry, dte: r.dte, underlying: r.underlying, iv_ratio: r.iv_ratio,
      sideDesc: `Sell ${r.side === "CE" ? "Call" : "Put"}`,
      positionDesc: `${num(r.short_strike, 0)}/${num(r.long_strike, 0)} BE ${num(r.breakeven, 0)}`,
      oiDesc: oiPair(r.oi_short, r.oi_long),
      credit: r.credit, max_profit: r.max_profit, max_risk: r.max_risk, lot_size: r.lot_size,
      max_profit_per_lot: r.max_profit_per_lot, max_risk_per_lot: r.max_risk_per_lot, ror_pct: r.ror_pct,
    })));
    const condorRows: MergedRow[] = (sellData?.top_picks ?? []).map((r, i) => ({
      strategy: "CONDOR" as const, tierId: "condor", tierLabel: "Condor", rank: i + 1,
      symbol: r.symbol, group: r.group, expiry: r.expiry, dte: r.dte, underlying: r.underlying, iv_ratio: r.iv_ratio,
      sideDesc: `${r.richer_side === "CE" ? "Call" : "Put"} richer`,
      positionDesc: `C ${num(r.short_ce, 0)}/${num(r.long_ce, 0)} · P ${num(r.short_pe, 0)}/${num(r.long_pe, 0)}`,
      oiDesc: `C ${oiPair(r.oi_short_ce, r.oi_long_ce)} · P ${oiPair(r.oi_short_pe, r.oi_long_pe)}`,
      credit: r.credit, max_profit: r.max_profit, max_risk: r.max_risk, lot_size: r.lot_size,
      max_profit_per_lot: r.max_profit_per_lot, max_risk_per_lot: r.max_risk_per_lot, ror_pct: r.ror_pct,
    }));
    let all = [...bwRows, ...skewRows, ...condorRows];
    if (filterStrategy) all = all.filter((r) => r.strategy === filterStrategy);
    // one stock, multiple strategies/tiers firing today -> keep only the most profitable (by
    // absolute max profit/lot, not ror_pct -- see maxProfitScore
    return dedupMostProfitable(all, (r) => r.symbol, maxProfitScore);
  }, [bwTiers, skewTiers, sellData, filterStrategy]);

  const histFlat: MergedHistRow[] = useMemo(() => {
    const bwRows: MergedHistRow[] = (bwHist?.rows ?? []).map((r) => ({
      strategy: "BW" as const, tierId: r.tier, symbol: r.symbol, group: r.group, signal_date: r.signal_date,
      expiry: r.expiry, dte: r.dte, iv_ratio: r.iv_ratio,
      positionDesc: `C ${num(r.short_ce, 0)}/${num(r.long_ce, 0)} · P ${num(r.short_pe, 0)}/${num(r.long_pe, 0)}`,
      oiDesc: `C ${oiPair(r.oi_short_ce, r.oi_long_ce)} · P ${oiPair(r.oi_short_pe, r.oi_long_pe)}`,
      credit: r.credit, max_profit: r.max_profit, max_risk: r.max_risk, lot_size: r.lot_size,
      max_risk_per_lot: r.max_risk_per_lot, max_profit_per_lot: r.max_profit_per_lot, entry_ror_pct: r.entry_ror_pct,
      exit_value: r.exit_value, pnl: r.pnl, pnl_per_lot: r.pnl_per_lot, ror_pct: r.ror_pct, max_dd_pct: r.max_dd_pct, outcome: r.outcome,
    }));
    const skewRows: MergedHistRow[] = (skewHist?.rows ?? []).map((r) => ({
      strategy: "SIV" as const, tierId: r.tier, symbol: r.symbol, group: r.group, signal_date: r.signal_date,
      expiry: r.expiry, dte: r.dte, iv_ratio: r.iv_ratio,
      positionDesc: `${num(r.short_strike, 0)}/${num(r.long_strike, 0)} (${r.side})`,
      oiDesc: oiPair(r.oi_short, r.oi_long),
      credit: r.credit, max_profit: r.max_profit, max_risk: r.max_risk, lot_size: r.lot_size,
      max_risk_per_lot: r.max_risk_per_lot, max_profit_per_lot: r.max_profit_per_lot, entry_ror_pct: r.entry_ror_pct,
      exit_value: r.exit_value, pnl: r.pnl, pnl_per_lot: r.pnl_per_lot, ror_pct: r.ror_pct, max_dd_pct: r.max_dd_pct, outcome: r.outcome,
    }));
    const condorRows: MergedHistRow[] = (sellHist?.rows ?? []).map((r) => ({
      strategy: "CONDOR" as const, tierId: "condor", symbol: r.symbol, group: r.group, signal_date: r.signal_date,
      expiry: r.expiry, dte: r.dte, iv_ratio: r.iv_ratio,
      positionDesc: `C ${num(r.short_ce, 0)}/${num(r.long_ce, 0)} · P ${num(r.short_pe, 0)}/${num(r.long_pe, 0)}`,
      oiDesc: `C ${oiPair(r.oi_short_ce, r.oi_long_ce)} · P ${oiPair(r.oi_short_pe, r.oi_long_pe)}`,
      credit: r.credit, max_profit: r.max_profit, max_risk: r.max_risk, lot_size: r.lot_size,
      max_risk_per_lot: r.max_risk_per_lot, max_profit_per_lot: r.max_profit_per_lot, entry_ror_pct: r.entry_ror_pct,
      exit_value: r.exit_value, pnl: r.pnl, pnl_per_lot: r.pnl_per_lot, ror_pct: r.ror_pct, max_dd_pct: r.max_dd_pct, outcome: r.outcome,
    }));
    let all = [...bwRows, ...skewRows, ...condorRows];
    if (filterStrategy) all = all.filter((r) => r.strategy === filterStrategy);
    // one stock, multiple strategies/tiers firing the same day -> keep only the most profitable
    // (by max profit/lot, same metric as the daily dedup -- see maxProfitScore)
    return dedupMostProfitable(all, (r) => `${r.symbol}|${r.signal_date}`, maxProfitScore);
  }, [bwHist, skewHist, sellHist, filterStrategy]);

  const allSyms = filterSym === "";
  const months = useMemo(() => monthsOf(histFlat, (r) => r.signal_date), [histFlat]);
  const histRowsAll = useMemo(() => (filterMonth ? histFlat.filter((r) => r.signal_date.startsWith(filterMonth)) : histFlat), [histFlat, filterMonth]);
  // histFlat is 3 strategies concatenated (each individually date-sorted by its own API call,
  // but the CONCATENATION isn't globally sorted) -- slicing it as-is for the "all months" cap
  // took an arbitrary bw-then-skew-then-condor prefix rather than the most recent rows, which
  // silently hid every recent Condor entry once bw+skew alone exceeded the cap. Sort by date
  // first so the cap actually keeps the most recent signals across all three strategies.
  const histRows = filterMonth ? histRowsAll
    : [...histRowsAll].sort((a, b) => b.signal_date.localeCompare(a.signal_date)).slice(0, MAX_UNFILTERED_HIST_ROWS);
  const hs = useMemo(() => {
    if (!histRows.length) return null;
    // exclude "pending" rows (forward window not complete yet, see gen_sell_signal_history.py)
    // from every ror_pct-derived stat -- their ror_pct is null, not zero
    const completed = histRows.filter((r) => r.outcome !== "pending");
    const rors = completed.map((r) => r.ror_pct as number);
    return {
      n: histRows.length, win_rate: completed.length ? completed.filter((r) => r.outcome === "win").length / completed.length : 0,
      median_ror_pct: rors.length ? Math.round(median(rors) * 10) / 10 : 0,
      worst_ror_pct: rors.length ? Math.round(Math.min(...rors) * 10) / 10 : 0,
      worst_dd_pct: Math.round(Math.min(...histRows.map((r) => r.max_dd_pct)) * 10) / 10,
    };
  }, [histRows]);
  const dailySort = useSortedRows(dailyFlat, "ror_pct" as never, "desc");
  const histSort = useSortedRows(histRows, "signal_date" as never, "desc");

  const allTierChips = [
    ...bwTiers.map((t) => ({ strategy: "BW" as const, id: t.id, label: t.label, fired_today: t.fired_today, backtest: t.backtest })),
    ...skewTiers.map((t) => ({ strategy: "SIV" as const, id: t.id, label: t.label, fired_today: t.fired_today, backtest: t.backtest })),
  ];

  return (
    <>
      <section className="panel cockpit">
        <div className="panel-title"><h2>Sell Signals — Condor + Broken-Wing + IV Skew (merged)</h2>
          <span>as of {bwData?.as_of ?? skewData?.as_of ?? sellData?.as_of ?? "—"}</span></div>
        <div className="sell-explain">
          <div className="sell-rule">
            <strong>Condor</strong> Sell the ~2% OTM call &amp; put, buy SYMMETRIC ~5% wings (loss always capped).
          </div>
          <div className="sell-rule">
            <strong>Broken-Wing Condor</strong> Sell the ~2% OTM call &amp; put, buy the wings ASYMMETRICALLY
            (narrow call wing / wide put wing) at 3 increasing asymmetry tiers. Max loss always capped at (wider wing − credit).
          </div>
          <div className="sell-rule">
            <strong>IV Skew</strong> Sell only the richer-IV side (CE or PE), buy a wing on that same side, at 3
            increasing wing-width tiers. <b>Server-side excluded when the symbol already has a live Broken-Wing signal
            that day</b> (Condor is not cross-excluded against either).
          </div>
          <div className="sell-rule">
            <strong>One signal per stock</strong> When more than one strategy/tier fires on the same stock the same day,
            only the single most profitable signal is kept here — the rest are hidden, not duplicated.
          </div>
          <div className="sell-rule"><strong>Exit</strong> Close at <b>~50% of max profit</b> or by expiry (whichever first). Same DTE-floor /
            entry-ror safety rules as before (9 days for Condor/Broken-Wing, 15 for Skew).</div>
        </div>
        <div className="bw-tier-status">
          {allTierChips.map((t) => (
            <div key={`${t.strategy}-${t.id}`} className={`bw-tier-chip ${t.fired_today ? "live" : ""}`}>
              <span className="bw-tier-dot" />
              <strong>{t.strategy === "BW" ? "BW" : "SIV"} · {t.label}</strong>
              <span>{t.fired_today ? "firing today" : "quiet today"}</span>
              <span className="hint">{(t.backtest.win_rate * 100).toFixed(0)}% win · ₹{t.backtest.median_pnl_per_lot.toLocaleString("en-IN")}/lot median · ~{t.backtest.per_year}/yr</span>
            </div>
          ))}
        </div>
        <div className="table-wrap">
          <table className="freeze-cols">
            <thead><tr>
              <SortTh className="fz1" label="Strategy" sortKey="strategy" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh className="fz2" label="Tier" sortKey="tierLabel" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh className="fz3" label="Symbol" sortKey="symbol" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh label="Exp / DTE" sortKey="dte" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh label="Spot" sortKey="underlying" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh label="IV rich" sortKey="iv_ratio" sort={dailySort.sort} onSort={dailySort.onSort} />
              <th>Side</th><th>Position</th><th>OI (lots)</th>
              <SortTh label="Credit" sortKey="credit" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh label="Lot size" sortKey="lot_size" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh label="Max profit/lot" sortKey="max_profit_per_lot" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh label="Max risk/lot" sortKey="max_risk_per_lot" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh label="Ret/risk" sortKey="ror_pct" sort={dailySort.sort} onSort={dailySort.onSort} />
            </tr></thead>
            <tbody>
              {dailySort.sorted.map((r) => (
                <tr key={`${r.strategy}-${r.tierId}-${r.symbol}`} className={r.rank === 1 ? "sell-live" : "sell-secondary"}>
                  <td className="fz1"><span className={`strategy-tag ${r.strategy}`}>{r.strategy}</span></td>
                  <td className="fz2"><span className="hint">{r.tierLabel} #{r.rank}</span></td>
                  <td className="fz3"><BWSymbol symbol={r.symbol} stats={symStats?.symbols[r.symbol]} onOpen={setOpenSymbol} /></td>
                  <td>{r.expiry.slice(5)} · {r.dte}d</td>
                  <td>{num(r.underlying, 0)}</td>
                  <td>{r.iv_ratio != null ? <span className={r.iv_ratio >= 1.1 ? "move-up" : "hint"}>{r.iv_ratio.toFixed(2)}×</span> : "—"}</td>
                  <td className="hint">{r.sideDesc}</td>
                  <td className="hint">{r.positionDesc}</td>
                  <td className="hint">{r.oiDesc}</td>
                  <td>{num(r.credit, 1)}</td>
                  <td>{r.lot_size ?? "—"}</td>
                  <td className="move-up">{r.max_profit_per_lot != null ? `₹${Math.round(r.max_profit_per_lot).toLocaleString("en-IN")}` : "—"}</td>
                  <td>{r.max_risk_per_lot != null ? `₹${Math.round(r.max_risk_per_lot).toLocaleString("en-IN")}` : "—"}</td>
                  <td><strong>{r.ror_pct.toFixed(0)}%</strong></td>
                </tr>
              ))}
              {!dailyFlat.length && <tr><td colSpan={13} className="empty-cell">No candidate clears the ret/risk bar in any tier today</td></tr>}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel cockpit" style={{ marginTop: 14 }}>
        <div className="panel-title">
          <h2>Signal history — returns per fired signal</h2>
          <div className="panel-title-controls">
            <label className="chk"><Coins size={15} />
              <select value={filterStrategy} onChange={(e) => setFilterStrategy(e.target.value as "" | "BW" | "SIV" | "CONDOR")} style={{ minWidth: 130 }}>
                <option value="">All strategies</option>
                <option value="CONDOR">Condor</option>
                <option value="BW">Broken-Wing</option>
                <option value="SIV">IV Skew</option>
              </select>
            </label>
            <label className="chk"><Coins size={15} />
              <select value={filterSym} onChange={(e) => setFilterSym(e.target.value)} style={{ minWidth: 160 }}>
                <option value="">All stocks</option>
                {symbols.map((s) => <option key={s.symbol} value={s.symbol}>{s.symbol} ({GROUP_LABEL[s.group] ?? s.group})</option>)}
              </select>
            </label>
            <MonthFilter months={months} value={filterMonth} onChange={setFilterMonth} />
          </div>
        </div>
        {hs ? (
          <div className="sell-bt">{hs.n} signals · win rate <b>{(hs.win_rate * 100).toFixed(0)}%</b> ·
            median <b>{hs.median_ror_pct >= 0 ? "+" : ""}{hs.median_ror_pct}%</b> return-on-risk ·
            worst trade <b>{hs.worst_ror_pct}%</b> · worst drawdown <b>{hs.worst_dd_pct}%</b> of risk
            {!filterMonth && histRowsAll.length > histRows.length ? <span className="hint"> · showing the most recent {histRows.length} of {histRowsAll.length} — pick a month to see more</span> : null}</div>
        ) : null}
        <div className="table-wrap">
          <table className="freeze-cols">
            <thead><tr>
              <SortTh className="fz1" label="Strategy" sortKey="strategy" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh className="fz2" label="Signal date" sortKey="signal_date" sort={histSort.sort} onSort={histSort.onSort} />
              {allSyms ? <SortTh className="fz3" label="Symbol" sortKey="symbol" sort={histSort.sort} onSort={histSort.onSort} /> : null}
              <th>Tier</th>
              <SortTh label="Exp / DTE" sortKey="dte" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh label="IV" sortKey="iv_ratio" sort={histSort.sort} onSort={histSort.onSort} />
              <th>Position</th>
              <th>OI (lots)</th>
              <SortTh label="Credit" sortKey="credit" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh label="Lot size" sortKey="lot_size" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh label="Max profit/lot" sortKey="max_profit_per_lot" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh label="Max risk/lot" sortKey="max_risk_per_lot" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh label="Exit value" sortKey="exit_value" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh label="PnL/lot" sortKey="pnl_per_lot" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh label="Ret/risk" sortKey="ror_pct" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh label="Max DD" sortKey="max_dd_pct" sort={histSort.sort} onSort={histSort.onSort} />
              <th></th>
            </tr></thead>
            <tbody>
              {histSort.sorted.map((r, i) => (
                <tr key={`${r.strategy}-${r.symbol}-${r.signal_date}-${i}`}>
                  <td className="fz1"><span className={`strategy-tag ${r.strategy}`}>{r.strategy}</span></td>
                  <td className="fz2">{r.signal_date}</td>
                  {allSyms ? <td className="fz3"><SymbolLink symbol={r.symbol} onOpen={setOpenSymbol} /> <span className="hint">{r.group.slice(0, 1)}</span></td> : null}
                  <td className="hint">{MERGED_TIER_LABEL[r.tierId] ?? r.tierId}</td>
                  <td>{r.expiry.slice(5)} · {r.dte}d</td>
                  <td>{r.iv_ratio != null ? `${r.iv_ratio.toFixed(2)}×` : "—"}</td>
                  <td className="hint">{r.positionDesc}</td>
                  <td className="hint">{r.oiDesc}</td>
                  <td>{num(r.credit, 1)}</td>
                  <td>{r.lot_size ?? "—"}</td>
                  <td className="move-up">{r.max_profit_per_lot != null ? `₹${Math.round(r.max_profit_per_lot).toLocaleString("en-IN")}` : "—"}</td>
                  <td>{r.max_risk_per_lot != null ? `₹${Math.round(r.max_risk_per_lot).toLocaleString("en-IN")}` : "—"}</td>
                  <td>{num(r.exit_value, 1)}</td>
                  <td className={r.pnl == null ? "hint" : r.pnl >= 0 ? "move-up" : "move-down"}>
                    {r.pnl_per_lot != null
                      ? <>{r.pnl_per_lot >= 0 ? "+" : ""}₹{Math.round(r.pnl_per_lot).toLocaleString("en-IN")}
                          <span className="hint"> ({r.pnl != null && r.pnl >= 0 ? "+" : ""}{num(r.pnl, 1)}/share)</span></>
                      : r.pnl == null ? "…" : <>{r.pnl >= 0 ? "+" : ""}{num(r.pnl, 1)}</>}
                  </td>
                  <td className={r.ror_pct == null ? "hint" : r.ror_pct >= 0 ? "move-up" : "move-down"}>{r.ror_pct == null ? "…" : <>{r.ror_pct >= 0 ? "+" : ""}{r.ror_pct.toFixed(0)}%</>}</td>
                  <td className="move-down">{r.max_dd_pct.toFixed(0)}%</td>
                  <td>{r.outcome === "win" ? "✓" : r.outcome === "pending" ? <span className="hint" title="forward window not complete yet">…</span> : "✕"}</td>
                </tr>
              ))}
              {!histRows.length && <tr><td colSpan={allSyms ? 17 : 16} className="empty-cell">No signals</td></tr>}
            </tbody>
          </table>
        </div>
      </section>
      {openSymbol && <ChartModal symbol={openSymbol} onClose={() => setOpenSymbol(null)} />}
    </>
  );
}

function SellStrategies() {
  const [data, setData] = useState<SellResp | null>(null);
  const [symbols, setSymbols] = useState<{ symbol: string; group: string }[]>([]);
  const [filterSym, setFilterSym] = useState("");   // "" = all
  const [filterMonth, setFilterMonth] = useState("");
  const [hist, setHist] = useState<SellHist | null>(null);
  const [openSymbol, setOpenSymbol] = useState<string | null>(null);
  useEffect(() => { getJson<SellResp>("/prod2/sell_strategies").then(setData).catch(() => setData(null)); }, []);
  useEffect(() => { getJson<{ symbol: string; group: string }[]>("/prod2/symbols").then(setSymbols).catch(() => {}); }, []);
  useEffect(() => {
    getJson<SellHist>(`/prod2/sell_signal_history${filterSym ? `?symbol=${filterSym}` : ""}`).then(setHist).catch(() => setHist(null));
  }, [filterSym]);
  useEffect(() => setFilterMonth(""), [filterSym]);

  const bt = data?.backtest;
  const daily = data?.top_picks ?? [];
  const allSyms = filterSym === "";
  const months = useMemo(() => monthsOf(hist?.rows ?? [], (r) => r.signal_date), [hist]);
  const histRowsAll = useMemo(() => {
    const all = hist?.rows ?? [];
    return filterMonth ? all.filter((r) => r.signal_date.startsWith(filterMonth)) : all;
  }, [hist, filterMonth]);
  const histRows = filterMonth ? histRowsAll : histRowsAll.slice(0, MAX_UNFILTERED_HIST_ROWS);
  const hs = useMemo(() => {
    if (!histRows.length) return null;
    // exclude "pending" rows (forward window not complete yet, see gen_sell_signal_history.py)
    // from every ror_pct/pnl-derived stat -- their ror_pct/pnl is null, not zero
    const completed = histRows.filter((r) => r.outcome !== "pending");
    const rors = completed.map((r) => r.ror_pct as number);
    return {
      n: histRows.length, win_rate: completed.length ? completed.filter((r) => r.outcome === "win").length / completed.length : 0,
      ev_ror_pct: rors.length ? Math.round((rors.reduce((s, v) => s + v, 0) / rors.length) * 10) / 10 : 0,
      median_ror_pct: rors.length ? Math.round(median(rors) * 10) / 10 : 0,
      worst_ror_pct: rors.length ? Math.round(Math.min(...rors) * 10) / 10 : 0,
      worst_dd_pct: Math.round(Math.min(...histRows.map((r) => r.max_dd_pct)) * 10) / 10,
      total_pnl: Math.round(completed.reduce((s, r) => s + (r.pnl as number), 0) * 10) / 10,
    };
  }, [histRows]);

  return (
    <>
      <section className="panel cockpit">
        <div className="panel-title"><h2>Sell Strategies — defined-risk iron condor</h2>
          <span>as of {data?.as_of ?? "—"} · 1 pick a day</span></div>
        <div className="sell-explain">
          <div className="sell-rule">
            <strong>Structure</strong> Sell the ~2% OTM call & put, buy the wings 5% further out (~7% OTM total) — a delta-neutral iron condor.
            Widened from a 3% wing after testing showed a similar win rate but meaningfully higher absolute rupee profit per lot.
            <b> Max loss is always capped</b> at (wing width − credit); you can never lose more than the max-risk shown.
          </div>
          <div className="sell-rule"><strong>Signal / entry</strong> THE gate is entry-time <b>credit/max-risk (ret/risk) &gt; 150%</b> — the theoretical
            max-profit/max-risk of the setup, knowable before the trade (unlike a realized outcome). Also requires <b>at least 9 days to expiry</b> —
            never closer, to stay clear of NSE's physical-delivery margin ramp (ITM margin escalates 10/25/45/70/100%+ of contract value starting
            4 trading days before expiry). <b>#1</b> is the single richest entry-window setup across both groups (the backtested rule, capped well
            under 10/week); <b>#2–#3</b> below are additional options, not part of that backtest. A position still open when DTE drops to 4 is
            force-closed regardless of profit target.</div>
          <div className="sell-rule"><strong>Exit</strong> Close at <b>~50% of max profit</b> or by expiry (whichever first); it decays in your favour while price stays between the breakevens.</div>
          {bt ? (
            <div className="sell-bt">Full gated pool ({bt.window}): <b>+{(bt.ev_on_risk * 100).toFixed(0)}%</b> mean return-on-risk,
              win rate <b>{(bt.win_rate * 100).toFixed(0)}%</b>, worst <b>{bt.worst}</b>.
              <span className="hint"> Gross of costs/STT — model these before sizing up.</span></div>
          ) : null}
          {bt?.best_of_day ? (
            <div className="sell-bt">This tab's #1 rule (1 pick/day, ~{bt.best_of_day.per_week}/week, n={bt.best_of_day.n}):
              <b> +{(bt.best_of_day.ev_on_risk * 100).toFixed(0)}%</b> mean return-on-risk,
              win rate <b>{(bt.best_of_day.win_rate * 100).toFixed(0)}%</b>, worst <b>{bt.best_of_day.worst}</b>
              {bt.best_of_day.median_pnl_per_lot != null ? <> · median <b>₹{bt.best_of_day.median_pnl_per_lot.toLocaleString("en-IN")}/lot</b> realized</> : null}.</div>
          ) : null}
        </div>
        <div className="table-wrap">
          <table>
            <thead><tr>
              <th></th><th>Symbol</th><th>Exp / DTE</th><th>Spot</th><th>IV rich</th><th>More profitable</th>
              <th>Short C / Long / BE</th><th>Short P / Long / BE</th><th>Credit</th><th>Max profit</th><th>Max risk</th>
              <th>Lot size</th><th>Max profit/lot</th><th>Max risk/lot</th><th>Ret/risk</th>
            </tr></thead>
            <tbody>
              {daily.map((r, i) => (
                <tr key={r.symbol} className={i === 0 ? "sell-live" : "sell-secondary"}>
                  <td><span className={`rank-badge ${i === 0 ? "primary" : ""}`}>#{i + 1}</span></td>
                  <td><SymbolLink symbol={r.symbol} onOpen={setOpenSymbol} /></td>
                  <td>{r.expiry.slice(5)} · {r.dte}d</td>
                  <td>{num(r.underlying, 0)}</td>
                  <td>{r.iv_ratio != null ? <span className={r.iv_ratio >= 1.1 ? "move-up" : "hint"}>{r.iv_ratio.toFixed(2)}×</span> : "—"}</td>
                  <td><span className={r.richer_side === "CE" ? "side long" : "side short"}>{r.richer_side === "CE" ? "Call" : "Put"} side</span>
                    <span className="hint"> ({num(Math.max(r.ce_credit, r.pe_credit), 1)} of {num(r.credit, 1)})</span></td>
                  <td>{num(r.short_ce, 0)}{r.oi_short_ce != null ? <span className="hint">({r.oi_short_ce})</span> : null} / {num(r.long_ce, 0)}{r.oi_long_ce != null ? <span className="hint">({r.oi_long_ce})</span> : null} / <span className="hint">{num(r.be_high, 0)}</span></td>
                  <td>{num(r.short_pe, 0)}{r.oi_short_pe != null ? <span className="hint">({r.oi_short_pe})</span> : null} / {num(r.long_pe, 0)}{r.oi_long_pe != null ? <span className="hint">({r.oi_long_pe})</span> : null} / <span className="hint">{num(r.be_low, 0)}</span></td>
                  <td>{num(r.credit, 1)} <span className="hint">(sell {num(r.sell_premium, 1)} / buy {num(r.buy_premium, 1)})</span></td>
                  <td className="move-up">{num(r.max_profit, 1)}</td>
                  <td>{num(r.max_risk, 1)} <span className="hint">(width {num(Math.max(r.long_ce - r.short_ce, r.short_pe - r.long_pe), 1)} / credit {num(r.credit, 1)})</span></td>
                  <td>{r.lot_size ?? "—"}</td>
                  <td className="move-up">{r.max_profit_per_lot != null ? `₹${Math.round(r.max_profit_per_lot).toLocaleString("en-IN")}` : "—"}</td>
                  <td>{r.max_risk_per_lot != null ? `₹${Math.round(r.max_risk_per_lot).toLocaleString("en-IN")}` : "—"}</td>
                  <td><strong>{r.ror_pct.toFixed(0)}%</strong></td>
                </tr>
              ))}
              {!daily.length && <tr><td colSpan={15} className="empty-cell">No candidate clears the 150% ret/risk bar today</td></tr>}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel cockpit" style={{ marginTop: 14 }}>
        <div className="panel-title">
          <h2>Signal history — returns per fired signal</h2>
          <div className="panel-title-controls">
            <label className="chk"><Coins size={15} />
              <select value={filterSym} onChange={(e) => setFilterSym(e.target.value)} style={{ minWidth: 160 }}>
                <option value="">All stocks</option>
                {symbols.map((s) => <option key={s.symbol} value={s.symbol}>{s.symbol} ({GROUP_LABEL[s.group] ?? s.group})</option>)}
              </select>
            </label>
            <MonthFilter months={months} value={filterMonth} onChange={setFilterMonth} />
          </div>
        </div>
        {hs ? (
          <div className="sell-bt">{hs.n} signals · win rate <b>{(hs.win_rate * 100).toFixed(0)}%</b> ·
            mean <b>{hs.ev_ror_pct >= 0 ? "+" : ""}{hs.ev_ror_pct}%</b> / median <b>{hs.median_ror_pct >= 0 ? "+" : ""}{hs.median_ror_pct}%</b> return-on-risk ·
            worst trade <b>{hs.worst_ror_pct}%</b> · worst drawdown <b>{hs.worst_dd_pct}%</b> of risk
            {!filterMonth && histRowsAll.length > histRows.length ? <span className="hint"> · showing the most recent {histRows.length} of {histRowsAll.length} — pick a month to see more</span> : null}</div>
        ) : null}
        <div className="table-wrap">
          <table>
            <thead><tr>
              <th>Signal date</th>{allSyms ? <th>Symbol</th> : null}<th>Exp / DTE</th><th>IV</th>
              <th>Short C / P</th><th>Credit</th><th>Max profit</th><th>Max risk</th><th>Lot size</th><th>Max risk/lot</th>
              <th>Exit value</th><th>PnL/lot</th><th>Ret/risk</th><th>Max DD</th><th></th>
            </tr></thead>
            <tbody>
              {histRows.map((r, i) => (
                <tr key={`${r.symbol}-${r.signal_date}-${i}`}>
                  <td>{r.signal_date}</td>
                  {allSyms ? <td><SymbolLink symbol={r.symbol} onOpen={setOpenSymbol} /> <span className="hint">{r.group.slice(0, 1)}</span></td> : null}
                  <td>{r.expiry.slice(5)} · {r.dte}d</td>
                  <td>{r.iv_ratio != null ? `${r.iv_ratio.toFixed(2)}×` : "—"}</td>
                  <td>{num(r.short_ce, 0)} / {num(r.short_pe, 0)}</td>
                  <td>{num(r.credit, 1)} <span className="hint">(sell {num(r.sell_premium, 1)} / buy {num(r.buy_premium, 1)})</span></td>
                  <td className="move-up">{num(r.max_profit, 1)} <span className="hint">{r.max_profit_per_lot != null ? `(₹${Math.round(r.max_profit_per_lot).toLocaleString("en-IN")}/lot)` : ""}</span></td>
                  <td>{num(r.max_risk, 1)} <span className="hint">(width {num(Math.max(r.long_ce - r.short_ce, r.short_pe - r.long_pe), 1)} / credit {num(r.credit, 1)})</span></td>
                  <td>{r.lot_size ?? "—"}</td>
                  <td>{r.max_risk_per_lot != null ? `₹${Math.round(r.max_risk_per_lot).toLocaleString("en-IN")}` : "—"}</td>
                  <td>{num(r.exit_value, 1)}</td>
                  <td className={r.pnl == null ? "hint" : r.pnl >= 0 ? "move-up" : "move-down"}>
                    {r.pnl_per_lot != null
                      ? <>{r.pnl_per_lot >= 0 ? "+" : ""}₹{Math.round(r.pnl_per_lot).toLocaleString("en-IN")}
                          <span className="hint"> ({r.pnl != null && r.pnl >= 0 ? "+" : ""}{num(r.pnl, 1)}/share)</span></>
                      : r.pnl == null ? "…" : <>{r.pnl >= 0 ? "+" : ""}{num(r.pnl, 1)}</>}
                  </td>
                  <td className={r.ror_pct == null ? "hint" : r.ror_pct >= 0 ? "move-up" : "move-down"}>{r.ror_pct == null ? "…" : <>{r.ror_pct >= 0 ? "+" : ""}{r.ror_pct.toFixed(0)}%</>}</td>
                  <td className="move-down">{r.max_dd_pct.toFixed(0)}%</td>
                  <td>{r.outcome === "win" ? "✓" : r.outcome === "pending" ? <span className="hint" title="forward window not complete yet">…</span> : "✕"}</td>
                </tr>
              ))}
              {!histRows.length && <tr><td colSpan={allSyms ? 16 : 15} className="empty-cell">No signals</td></tr>}
            </tbody>
          </table>
        </div>
      </section>
      {openSymbol && <ChartModal symbol={openSymbol} onClose={() => setOpenSymbol(null)} />}
    </>
  );
}

// ----------------------------------------------------------------- Broken-Wing Condor (3 tiers, asymmetric wings)
type BWRow = {
  symbol: string; group: string; expiry: string; dte: number; underlying: number; iv_ratio: number | null;
  short_ce: number; long_ce: number; short_pe: number; long_pe: number;
  call_width_pct: number; put_width_pct: number;
  oi_short_ce: number | null; oi_long_ce: number | null; oi_short_pe: number | null; oi_long_pe: number | null;
  sell_premium: number; buy_premium: number;
  credit: number; max_risk: number; lot_size: number | null; max_risk_per_lot: number | null;
  max_profit: number; max_profit_per_lot: number | null;
  ror_pct: number; be_low: number; be_high: number;
  richer_side: "CE" | "PE"; ce_credit: number; pe_credit: number; in_window: boolean;
};
type BWTierBacktest = { window: string; win_rate: number; median_pnl_per_lot: number; median_pnl_per_lot_per_day: number;
                         per_year: number; pct_trades_over_10k: number; note: string };
type BWTier = {
  id: string; label: string; wing_ce: number; wing_pe: number; fired_today: boolean;
  backtest: BWTierBacktest; candidates: BWRow[]; top_picks: BWRow[];
};
type BWResp = { as_of: string | null; params: Record<string, number>; tiers: BWTier[] };
type BWHistRow = {
  tier: string; symbol: string; group: string; signal_date: string; expiry: string; dte: number; iv_ratio: number | null;
  short_ce: number; long_ce: number; short_pe: number; long_pe: number;
  call_width_pct: number; put_width_pct: number;
  oi_short_ce: number | null; oi_long_ce: number | null; oi_short_pe: number | null; oi_long_pe: number | null;
  sell_premium: number; buy_premium: number;
  credit: number; max_risk: number; max_profit: number;
  lot_size: number | null; max_risk_per_lot: number | null; max_profit_per_lot: number | null;
  entry_ror_pct: number;   // theoretical ROR knowable at entry -- used to rank/dedup still-"pending" rows, which have no realized ror_pct yet
  exit_value: number; pnl: number | null; pnl_per_lot: number | null; ror_pct: number | null; max_dd_pct: number; outcome: string;   // outcome may be "pending" -- forward window not complete yet, see gen_sell_signal_history.py
};
type BWHist = {
  rows: BWHistRow[];
  summary: { n: number; n_pending?: number; win_rate: number; ev_ror_pct: number; median_ror_pct: number; worst_ror_pct: number; worst_dd_pct: number; total_pnl: number } | null;
};
const BW_TIER_LABEL: Record<string, string> = { t1_2x6: "Tier 1 · 2%/6%", t2_3x8: "Tier 2 · 3%/8%", t3_2x10: "Tier 3 · 2%/10%" };
type BWSymbolStatEntry = { n: number; median_pnl_per_lot: number; mean_pnl_per_lot: number; win_rate: number;
  last10: { entry_date: string; pnl_per_lot: number; ror_pct: number; outcome: string; tier: string }[] };
type BWSymbolStats = { symbols: Record<string, BWSymbolStatEntry> };
// Highlight threshold: only color a symbol once it has >=3 signals (avoids overclaiming on thin
// samples) AND its historical median PnL/lot clears a "worth noting" bar.
const BW_HIGHLIGHT_MEDIAN = 6000;
function BWSymbol({ symbol, stats, onOpen }: { symbol: string; stats: BWSymbolStatEntry | undefined; onOpen: (s: string) => void }) {
  const strong = stats && stats.n >= 3 && stats.median_pnl_per_lot >= BW_HIGHLIGHT_MEDIAN;
  const title = stats
    ? `${symbol} — ${stats.n} signals, median ₹${Math.round(stats.median_pnl_per_lot).toLocaleString("en-IN")}/lot, ${(stats.win_rate * 100).toFixed(0)}% win\nLast ${stats.last10.length}:\n` +
      stats.last10.map((r) => `${r.entry_date}  ${r.outcome === "win" ? "+" : ""}₹${Math.round(r.pnl_per_lot).toLocaleString("en-IN")}/lot  (${r.ror_pct >= 0 ? "+" : ""}${r.ror_pct.toFixed(0)}%, ${BW_TIER_LABEL[r.tier] ?? r.tier})`).join("\n")
    : undefined;
  return (
    <span title={title} className={strong ? "bw-symbol-strong" : undefined}>
      <SymbolLink symbol={symbol} onOpen={onOpen} />
    </span>
  );
}

function BrokenWingStrategy() {
  const [data, setData] = useState<BWResp | null>(null);
  const [symbols, setSymbols] = useState<{ symbol: string; group: string }[]>([]);
  const [filterSym, setFilterSym] = useState("");
  const [filterTier, setFilterTier] = useState("");
  const [filterMonth, setFilterMonth] = useState("");
  const [hist, setHist] = useState<BWHist | null>(null);
  const [openSymbol, setOpenSymbol] = useState<string | null>(null);
  const [symStats, setSymStats] = useState<BWSymbolStats | null>(null);
  useEffect(() => { getJson<BWResp>("/prod2/broken_wing_strategy").then(setData).catch(() => setData(null)); }, []);
  useEffect(() => { getJson<{ symbol: string; group: string }[]>("/prod2/symbols").then(setSymbols).catch(() => {}); }, []);
  useEffect(() => { getJson<BWSymbolStats>("/prod2/broken_wing_symbol_stats").then(setSymStats).catch(() => setSymStats(null)); }, []);
  useEffect(() => {
    const q = [filterSym && `symbol=${filterSym}`, filterTier && `tier=${filterTier}`].filter(Boolean).join("&");
    getJson<BWHist>(`/prod2/broken_wing_signal_history${q ? `?${q}` : ""}`).then(setHist).catch(() => setHist(null));
  }, [filterSym, filterTier]);
  useEffect(() => setFilterMonth(""), [filterSym, filterTier]);

  const tiers = data?.tiers ?? [];
  const allSyms = filterSym === "";
  const months = useMemo(() => monthsOf(hist?.rows ?? [], (r) => r.signal_date), [hist]);
  const histRowsAll = useMemo(() => {
    const all = hist?.rows ?? [];
    return filterMonth ? all.filter((r) => r.signal_date.startsWith(filterMonth)) : all;
  }, [hist, filterMonth]);
  const histRows = filterMonth ? histRowsAll : histRowsAll.slice(0, MAX_UNFILTERED_HIST_ROWS);
  const dailyFlat = useMemo(() => tiers.flatMap((t) => t.top_picks.map((r, i) => ({ ...r, tierId: t.id, tierLabel: t.label, rank: i + 1 }))), [tiers]);
  const dailySort = useSortedRows(dailyFlat, "ror_pct" as never, "desc");
  const histSort = useSortedRows(histRows, "signal_date" as never, "desc");
  const hs = useMemo(() => {
    if (!histRows.length) return null;
    // exclude "pending" rows (forward window not complete yet, see gen_sell_signal_history.py)
    // from every ror_pct/pnl-derived stat -- their ror_pct/pnl is null, not zero
    const completed = histRows.filter((r) => r.outcome !== "pending");
    const rors = completed.map((r) => r.ror_pct as number);
    return {
      n: histRows.length, win_rate: completed.length ? completed.filter((r) => r.outcome === "win").length / completed.length : 0,
      ev_ror_pct: rors.length ? Math.round((rors.reduce((s, v) => s + v, 0) / rors.length) * 10) / 10 : 0,
      median_ror_pct: rors.length ? Math.round(median(rors) * 10) / 10 : 0,
      worst_ror_pct: rors.length ? Math.round(Math.min(...rors) * 10) / 10 : 0,
      worst_dd_pct: Math.round(Math.min(...histRows.map((r) => r.max_dd_pct)) * 10) / 10,
      total_pnl: Math.round(completed.reduce((s, r) => s + (r.pnl as number), 0) * 10) / 10,
    };
  }, [histRows]);

  return (
    <>
      <section className="panel cockpit">
        <div className="panel-title"><h2>Broken-Wing Condor — asymmetric defined-risk, 3 tiers</h2>
          <span>as of {data?.as_of ?? "—"}</span></div>
        <div className="sell-explain">
          <div className="sell-rule">
            <strong>Structure</strong> Sell the ~2% OTM call &amp; put like the regular condor, but buy the wings ASYMMETRICALLY —
            always a narrow call wing / wide put wing, at three increasing levels of asymmetry (tiers below).
            <b> Max loss is always capped</b> at (wider wing − credit) — only one side can be breached at expiry, so risk is
            set by the wider (put) wing alone, not the sum of both. Indian equity/index options carry a persistent put skew
            (crash premium): the wide put wing buys cheap far-OTM protection while banking most of that rich put premium as
            credit; the narrow call wing still collects decent credit since calls aren't as richly priced.
          </div>
          <div className="sell-rule"><strong>Tiers — all live simultaneously</strong> Tier 1 (2%/6%) is the everyday base signal
            (fires most often, ~143/yr). Tier 2 (3%/8%) and Tier 3 (2%/10%) are progressively rarer and higher-quality
            (~51/yr and ~12/yr). <b>When Tier 2 or 3 also fires alongside Tier 1 on a given day, that's a genuine
            higher-conviction setup</b> — size accordingly. ~206 signals/yr combined (~4/week).</div>
          <div className="sell-rule"><strong>Universe &amp; entry gate</strong> Scanned universe is the static A/B stock
            groups plus each day's dynamic top-50-by-open-interest-in-lots stocks (not raw share OI, which is dominated
            by cheap high-share-count names — OI-in-lots correctly surfaces mega-caps). The entry gate is
            <b> stratified</b>: mega-caps need credit/max-risk &gt; 120% (they're structurally lower-IV/steadier, so a
            uniform bar under-represented them), everyone else needs &gt; 150%. At least 9 days to expiry either way.
            Does <b>not</b> exclude F&amp;O-ban-listed stocks (no ban-list data source available) — cross-check live picks
            against NSE's published ban list before trading.</div>
          <div className="sell-rule"><strong>Exit</strong> Close at <b>~50% of max profit</b> or by expiry (whichever first).</div>
        </div>
        <div className="bw-tier-status">
          {tiers.map((t) => (
            <div key={t.id} className={`bw-tier-chip ${t.fired_today ? "live" : ""}`}>
              <span className="bw-tier-dot" />
              <strong>{t.label}</strong>
              <span>{t.fired_today ? "firing today" : "quiet today"}</span>
              <span className="hint">{(t.backtest.win_rate * 100).toFixed(0)}% win · ₹{t.backtest.median_pnl_per_lot.toLocaleString("en-IN")}/lot median · ~{t.backtest.per_year}/yr</span>
            </div>
          ))}
        </div>
        <div className="table-wrap">
          <table className="freeze-cols">
            <thead><tr>
              <SortTh className="fz1" label="Tier" sortKey="tierId" sort={dailySort.sort} onSort={dailySort.onSort} />
              <th className="fz2"></th>
              <SortTh className="fz3" label="Symbol" sortKey="symbol" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh label="Exp / DTE" sortKey="dte" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh label="Spot" sortKey="underlying" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh label="IV rich" sortKey="iv_ratio" sort={dailySort.sort} onSort={dailySort.onSort} />
              <th>More profitable</th>
              <th>Short C / Long / BE</th><th>Short P / Long / BE</th><th>Wings</th>
              <SortTh label="Credit" sortKey="credit" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh label="Max profit" sortKey="max_profit" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh label="Max risk" sortKey="max_risk" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh label="Lot size" sortKey="lot_size" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh label="Max profit/lot" sortKey="max_profit_per_lot" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh label="Max risk/lot" sortKey="max_risk_per_lot" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh label="Ret/risk" sortKey="ror_pct" sort={dailySort.sort} onSort={dailySort.onSort} />
            </tr></thead>
            <tbody>
              {dailySort.sorted.map((r) => (
                <tr key={`${r.tierId}-${r.symbol}`} className={r.rank === 1 ? "sell-live" : "sell-secondary"}>
                  <td className="fz1"><span className="hint">{r.tierLabel}</span></td>
                  <td className="fz2"><span className={`rank-badge ${r.rank === 1 ? "primary" : ""}`}>#{r.rank}</span></td>
                  <td className="fz3"><BWSymbol symbol={r.symbol} stats={symStats?.symbols[r.symbol]} onOpen={setOpenSymbol} /></td>
                  <td>{r.expiry.slice(5)} · {r.dte}d</td>
                  <td>{num(r.underlying, 0)}</td>
                  <td>{r.iv_ratio != null ? <span className={r.iv_ratio >= 1.1 ? "move-up" : "hint"}>{r.iv_ratio.toFixed(2)}×</span> : "—"}</td>
                  <td><span className={r.richer_side === "CE" ? "side long" : "side short"}>{r.richer_side === "CE" ? "Call" : "Put"} side</span>
                    <span className="hint"> ({num(Math.max(r.ce_credit, r.pe_credit), 1)} of {num(r.credit, 1)})</span></td>
                  <td>{num(r.short_ce, 0)}{r.oi_short_ce != null ? <span className="hint">({r.oi_short_ce})</span> : null} / {num(r.long_ce, 0)}{r.oi_long_ce != null ? <span className="hint">({r.oi_long_ce})</span> : null} / <span className="hint">{num(r.be_high, 0)}</span></td>
                  <td>{num(r.short_pe, 0)}{r.oi_short_pe != null ? <span className="hint">({r.oi_short_pe})</span> : null} / {num(r.long_pe, 0)}{r.oi_long_pe != null ? <span className="hint">({r.oi_long_pe})</span> : null} / <span className="hint">{num(r.be_low, 0)}</span></td>
                  <td className="hint">{r.call_width_pct.toFixed(1)}% / {r.put_width_pct.toFixed(1)}%</td>
                  <td>{num(r.credit, 1)} <span className="hint">(sell {num(r.sell_premium, 1)} / buy {num(r.buy_premium, 1)})</span></td>
                  <td className="move-up">{num(r.max_profit, 1)}</td>
                  <td>{num(r.max_risk, 1)} <span className="hint">(width {num(Math.max(r.long_ce - r.short_ce, r.short_pe - r.long_pe), 1)} / credit {num(r.credit, 1)})</span></td>
                  <td>{r.lot_size ?? "—"}</td>
                  <td className="move-up">{r.max_profit_per_lot != null ? `₹${Math.round(r.max_profit_per_lot).toLocaleString("en-IN")}` : "—"}</td>
                  <td>{r.max_risk_per_lot != null ? `₹${Math.round(r.max_risk_per_lot).toLocaleString("en-IN")}` : "—"}</td>
                  <td><strong>{r.ror_pct.toFixed(0)}%</strong></td>
                </tr>
              ))}
              {!dailyFlat.length && <tr><td colSpan={17} className="empty-cell">No candidate clears the ret/risk bar in any tier today</td></tr>}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel cockpit" style={{ marginTop: 14 }}>
        <div className="panel-title">
          <h2>Signal history — returns per fired signal</h2>
          <div className="panel-title-controls">
            <label className="chk"><Coins size={15} />
              <select value={filterTier} onChange={(e) => setFilterTier(e.target.value)} style={{ minWidth: 140 }}>
                <option value="">All tiers</option>
                {Object.entries(BW_TIER_LABEL).map(([id, label]) => <option key={id} value={id}>{label}</option>)}
              </select>
            </label>
            <label className="chk"><Coins size={15} />
              <select value={filterSym} onChange={(e) => setFilterSym(e.target.value)} style={{ minWidth: 160 }}>
                <option value="">All stocks</option>
                {symbols.map((s) => <option key={s.symbol} value={s.symbol}>{s.symbol} ({GROUP_LABEL[s.group] ?? s.group})</option>)}
              </select>
            </label>
            <MonthFilter months={months} value={filterMonth} onChange={setFilterMonth} />
          </div>
        </div>
        {hs ? (
          <div className="sell-bt">{hs.n} signals · win rate <b>{(hs.win_rate * 100).toFixed(0)}%</b> ·
            mean <b>{hs.ev_ror_pct >= 0 ? "+" : ""}{hs.ev_ror_pct}%</b> / median <b>{hs.median_ror_pct >= 0 ? "+" : ""}{hs.median_ror_pct}%</b> return-on-risk ·
            worst trade <b>{hs.worst_ror_pct}%</b> · worst drawdown <b>{hs.worst_dd_pct}%</b> of risk
            {!filterMonth && histRowsAll.length > histRows.length ? <span className="hint"> · showing the most recent {histRows.length} of {histRowsAll.length} — pick a month to see more</span> : null}</div>
        ) : null}
        <div className="table-wrap">
          <table className="freeze-cols">
            <thead><tr>
              <SortTh className="fz1" label="Tier" sortKey="tier" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh className="fz2" label="Signal date" sortKey="signal_date" sort={histSort.sort} onSort={histSort.onSort} />
              {allSyms ? <SortTh className="fz3" label="Symbol" sortKey="symbol" sort={histSort.sort} onSort={histSort.onSort} /> : null}
              <SortTh label="Exp / DTE" sortKey="dte" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh label="IV" sortKey="iv_ratio" sort={histSort.sort} onSort={histSort.onSort} />
              <th>Short C / P</th>
              <th>Wings</th>
              <SortTh label="Credit" sortKey="credit" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh label="Max profit" sortKey="max_profit" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh label="Max risk" sortKey="max_risk" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh label="Lot size" sortKey="lot_size" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh label="Max risk/lot" sortKey="max_risk_per_lot" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh label="Exit value" sortKey="exit_value" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh label="PnL/lot" sortKey="pnl_per_lot" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh label="Ret/risk" sortKey="ror_pct" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh label="Max DD" sortKey="max_dd_pct" sort={histSort.sort} onSort={histSort.onSort} />
              <th></th>
            </tr></thead>
            <tbody>
              {histSort.sorted.map((r, i) => (
                <tr key={`${r.symbol}-${r.signal_date}-${i}`}>
                  <td className="fz1 hint">{BW_TIER_LABEL[r.tier] ?? r.tier}</td>
                  <td className="fz2">{r.signal_date}</td>
                  {allSyms ? <td className="fz3"><SymbolLink symbol={r.symbol} onOpen={setOpenSymbol} /> <span className="hint">{r.group.slice(0, 1)}</span></td> : null}
                  <td>{r.expiry.slice(5)} · {r.dte}d</td>
                  <td>{r.iv_ratio != null ? `${r.iv_ratio.toFixed(2)}×` : "—"}</td>
                  <td>{num(r.short_ce, 0)} / {num(r.short_pe, 0)}</td>
                  <td className="hint">{r.call_width_pct.toFixed(1)}% / {r.put_width_pct.toFixed(1)}%</td>
                  <td>{num(r.credit, 1)} <span className="hint">(sell {num(r.sell_premium, 1)} / buy {num(r.buy_premium, 1)})</span></td>
                  <td className="move-up">{num(r.max_profit, 1)} <span className="hint">{r.max_profit_per_lot != null ? `(₹${Math.round(r.max_profit_per_lot).toLocaleString("en-IN")}/lot)` : ""}</span></td>
                  <td>{num(r.max_risk, 1)} <span className="hint">(width {num(Math.max(r.long_ce - r.short_ce, r.short_pe - r.long_pe), 1)} / credit {num(r.credit, 1)})</span></td>
                  <td>{r.lot_size ?? "—"}</td>
                  <td>{r.max_risk_per_lot != null ? `₹${Math.round(r.max_risk_per_lot).toLocaleString("en-IN")}` : "—"}</td>
                  <td>{num(r.exit_value, 1)}</td>
                  <td className={r.pnl == null ? "hint" : r.pnl >= 0 ? "move-up" : "move-down"}>
                    {r.pnl_per_lot != null
                      ? <>{r.pnl_per_lot >= 0 ? "+" : ""}₹{Math.round(r.pnl_per_lot).toLocaleString("en-IN")}
                          <span className="hint"> ({r.pnl != null && r.pnl >= 0 ? "+" : ""}{num(r.pnl, 1)}/share)</span></>
                      : r.pnl == null ? "…" : <>{r.pnl >= 0 ? "+" : ""}{num(r.pnl, 1)}</>}
                  </td>
                  <td className={r.ror_pct == null ? "hint" : r.ror_pct >= 0 ? "move-up" : "move-down"}>{r.ror_pct == null ? "…" : <>{r.ror_pct >= 0 ? "+" : ""}{r.ror_pct.toFixed(0)}%</>}</td>
                  <td className="move-down">{r.max_dd_pct.toFixed(0)}%</td>
                  <td>{r.outcome === "win" ? "✓" : r.outcome === "pending" ? <span className="hint" title="forward window not complete yet">…</span> : "✕"}</td>
                </tr>
              ))}
              {!histRows.length && <tr><td colSpan={allSyms ? 17 : 16} className="empty-cell">No signals</td></tr>}
            </tbody>
          </table>
        </div>
      </section>
      {openSymbol && <ChartModal symbol={openSymbol} onClose={() => setOpenSymbol(null)} />}
    </>
  );
}

// Tiered skew response shape (mirrors BWTier/BWResp, but with skew's row shape)
type SkewTierRow = {
  symbol: string; group: string; expiry: string; dte: number; underlying: number; iv_ratio: number | null;
  side: "CE" | "PE"; ce_iv: number; pe_iv: number; skew: number;
  short_strike: number; long_strike: number;
  oi_short: number | null; oi_long: number | null;
  sell_premium: number; buy_premium: number;
  credit: number; max_risk: number; lot_size: number | null; max_risk_per_lot: number | null;
  max_profit: number; max_profit_per_lot: number | null;
  ror_pct: number; breakeven: number; in_window: boolean;
};
type SkewTierBacktest = { window: string; win_rate: number; median_pnl_per_lot: number; per_year: number; pct_trades_over_10k: number; note: string };
type SkewTier = { id: string; label: string; wing: number; fired_today: boolean; backtest: SkewTierBacktest; candidates: SkewTierRow[]; top_picks: SkewTierRow[] };
type SkewTieredResp = { as_of: string | null; params: Record<string, number>; excluded_by_broken_wing: string[]; tiers: SkewTier[] };
type SkewTieredHistRow = {
  tier: string; symbol: string; group: string; signal_date: string; expiry: string; dte: number; iv_ratio: number | null;
  side: "CE" | "PE"; ce_iv: number; pe_iv: number; skew: number; short_strike: number; long_strike: number;
  oi_short: number | null; oi_long: number | null;
  sell_premium: number; buy_premium: number;
  credit: number; max_risk: number; max_profit: number;
  lot_size: number | null; max_risk_per_lot: number | null; max_profit_per_lot: number | null;
  entry_ror_pct: number;   // theoretical ROR knowable at entry -- used to rank/dedup still-"pending" rows, which have no realized ror_pct yet
  exit_value: number; pnl: number | null; pnl_per_lot: number | null; ror_pct: number | null; max_dd_pct: number; outcome: string;   // outcome may be "pending" -- forward window not complete yet, see gen_sell_signal_history.py
};
type SkewTieredHist = {
  rows: SkewTieredHistRow[];
  summary: { n: number; n_pending?: number; win_rate: number; ev_ror_pct: number; median_ror_pct: number; worst_ror_pct: number; worst_dd_pct: number; total_pnl: number } | null;
};

// ----------------------------------------------------------------- Skew Strategy (directional leg) -- LEGACY, unused (merged into MergedSellSignals above)
type SkewRow = {
  symbol: string; group: string; expiry: string; dte: number; underlying: number; iv_ratio: number | null;
  side: "CE" | "PE"; ce_iv: number; pe_iv: number; skew: number;
  short_strike: number; long_strike: number; sell_premium: number; buy_premium: number;
  credit: number; max_risk: number; lot_size: number | null; max_risk_per_lot: number | null;
  max_profit: number; max_profit_per_lot: number | null;
  ror_pct: number; breakeven: number; in_window: boolean;
};
type SkewResp = {
  as_of: string | null; params: Record<string, number>;
  backtest: { window: string; ev_on_risk: number; win_rate: number; worst: string; note: string };
  candidates: SkewRow[]; top_picks: SkewRow[];
};
type SkewHistRow = {
  symbol: string; group: string; signal_date: string; expiry: string; dte: number; iv_ratio: number | null;
  side: "CE" | "PE"; ce_iv: number; pe_iv: number; skew: number; short_strike: number; long_strike: number;
  sell_premium: number; buy_premium: number;
  credit: number; max_risk: number; max_profit: number;
  lot_size: number | null; max_risk_per_lot: number | null; max_profit_per_lot: number | null;
  entry_ror_pct: number;   // theoretical ROR knowable at entry -- used to rank/dedup still-"pending" rows, which have no realized ror_pct yet
  exit_value: number; pnl: number | null; pnl_per_lot: number | null; ror_pct: number | null; max_dd_pct: number; outcome: string;   // outcome may be "pending" -- forward window not complete yet, see gen_sell_signal_history.py
};
type SkewHist = {
  rows: SkewHistRow[];
  summary: { n: number; n_pending?: number; win_rate: number; ev_ror_pct: number; median_ror_pct: number; worst_ror_pct: number; worst_dd_pct: number; total_pnl: number } | null;
};

function SkewStrategy() {
  const [data, setData] = useState<SkewResp | null>(null);
  const [symbols, setSymbols] = useState<{ symbol: string; group: string }[]>([]);
  const [filterSym, setFilterSym] = useState("");
  const [filterMonth, setFilterMonth] = useState("");
  const [hist, setHist] = useState<SkewHist | null>(null);
  const [openSymbol, setOpenSymbol] = useState<string | null>(null);
  useEffect(() => { getJson<SkewResp>("/prod2/skew_strategy").then(setData).catch(() => setData(null)); }, []);
  useEffect(() => { getJson<{ symbol: string; group: string }[]>("/prod2/symbols").then(setSymbols).catch(() => {}); }, []);
  useEffect(() => {
    getJson<SkewHist>(`/prod2/skew_signal_history${filterSym ? `?symbol=${filterSym}` : ""}`).then(setHist).catch(() => setHist(null));
  }, [filterSym]);
  useEffect(() => setFilterMonth(""), [filterSym]);

  const bt = data?.backtest;
  const daily = data?.top_picks ?? [];
  const allSyms = filterSym === "";
  const months = useMemo(() => monthsOf(hist?.rows ?? [], (r) => r.signal_date), [hist]);
  const histRowsAll = useMemo(() => {
    const all = hist?.rows ?? [];
    return filterMonth ? all.filter((r) => r.signal_date.startsWith(filterMonth)) : all;
  }, [hist, filterMonth]);
  const histRows = filterMonth ? histRowsAll : histRowsAll.slice(0, MAX_UNFILTERED_HIST_ROWS);
  const hs = useMemo(() => {
    if (!histRows.length) return null;
    // exclude "pending" rows (forward window not complete yet, see gen_sell_signal_history.py)
    // from every ror_pct/pnl-derived stat -- their ror_pct/pnl is null, not zero
    const completed = histRows.filter((r) => r.outcome !== "pending");
    const rors = completed.map((r) => r.ror_pct as number);
    return {
      n: histRows.length, win_rate: completed.length ? completed.filter((r) => r.outcome === "win").length / completed.length : 0,
      ev_ror_pct: rors.length ? Math.round((rors.reduce((s, v) => s + v, 0) / rors.length) * 10) / 10 : 0,
      median_ror_pct: rors.length ? Math.round(median(rors) * 10) / 10 : 0,
      worst_ror_pct: rors.length ? Math.round(Math.min(...rors) * 10) / 10 : 0,
      worst_dd_pct: Math.round(Math.min(...histRows.map((r) => r.max_dd_pct)) * 10) / 10,
      total_pnl: Math.round(completed.reduce((s, r) => s + (r.pnl as number), 0) * 10) / 10,
    };
  }, [histRows]);
  const dailySort = useSortedRows(daily, "ror_pct" as never, "desc");
  const histSort = useSortedRows(histRows, "signal_date" as never, "desc");

  return (
    <>
      <section className="panel cockpit" style={{ marginTop: 14 }}>
        <div className="panel-title"><h2>Directional credit spread — IV skew</h2>
          <span>as of {data?.as_of ?? "—"} · top 3 per group</span></div>
        <div className="sell-explain">
          <div className="sell-rule">
            <strong>Structure</strong> Sell only the side (call or put) whose ~2% OTM strike is priced with the <b>richer</b> Black-Scholes implied vol
            (skew = CE-IV − PE-IV), buy the +5% wing on that same side — a single-leg defined-risk credit spread.
            <b> Max loss is always capped</b> at (wing width − credit).
            <span className="hint"> This is a relative-value read on option pricing, not a forecast of which way the stock moves — direction alone is ≈ coin-flip on this universe.</span>
          </div>
          <div className="sell-rule"><strong>Signal / entry</strong> THE gate is entry-time <b>credit/max-risk (ret/risk) &gt; 150%</b>. Also requires
            <b> at least 15 days to expiry</b> — unlike the condor, this structure backtests <i>better</i> further from expiry, not closer, and it still
            stays clear of NSE's physical-delivery margin ramp. Rows meeting both are the <b>entry-window</b> picks (highlighted), ranked by ret/risk,
            top 3 per group shown; a position still open when DTE drops to 4 is force-closed.</div>
          <div className="sell-rule"><strong>Exit</strong> Close at <b>~50% of max profit</b> or by expiry (whichever first) — <b>no interim stop-loss</b>.
            Day-close dips over the hold tend to mean-revert, so a stop just locks in a temporary drawdown (finding carried over from an earlier backtest; not yet re-tested for this rule).</div>
          {bt ? (
            <div className="sell-bt">Backtest ({bt.window}): <b>+{(bt.ev_on_risk * 100).toFixed(0)}%</b> mean return-on-risk,
              win rate <b>{(bt.win_rate * 100).toFixed(0)}%</b>, worst <b>{bt.worst}</b>.
              <span className="hint"> Gross of costs/STT — model these before sizing up.</span></div>
          ) : null}
          {bt?.note ? <div className="sell-bt" style={{ borderColor: "#d8a13a" }}><b>⚠ Unverified:</b> {bt.note}</div> : null}
        </div>
        <div className="table-wrap">
          <table className="freeze-cols">
            <thead><tr>
              <SortTh className="fz1" label="Symbol" sortKey="symbol" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh label="Exp / DTE" sortKey="dte" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh label="Spot" sortKey="underlying" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh label="IV rich" sortKey="iv_ratio" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh label="Sell" sortKey="side" sort={dailySort.sort} onSort={dailySort.onSort} />
              <th>CE-IV / PE-IV</th><th>Short / Long</th>
              <SortTh label="Credit" sortKey="credit" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh label="Max profit" sortKey="max_profit" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh label="Max risk" sortKey="max_risk" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh label="Lot size" sortKey="lot_size" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh label="Max profit/lot" sortKey="max_profit_per_lot" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh label="Max risk/lot" sortKey="max_risk_per_lot" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh label="Ret/risk" sortKey="ror_pct" sort={dailySort.sort} onSort={dailySort.onSort} />
              <SortTh label="Breakeven" sortKey="breakeven" sort={dailySort.sort} onSort={dailySort.onSort} />
            </tr></thead>
            <tbody>
              {dailySort.sorted.map((r) => (
                <tr key={r.symbol} className={r.in_window ? "sell-live" : ""}>
                  <td className="fz1"><SymbolLink symbol={r.symbol} onOpen={setOpenSymbol} /></td>
                  <td>{r.expiry.slice(5)} · {r.dte}d</td>
                  <td>{num(r.underlying, 0)}</td>
                  <td>{r.iv_ratio != null ? <span className={r.iv_ratio >= 1.1 ? "move-up" : "hint"}>{r.iv_ratio.toFixed(2)}×</span> : "—"}</td>
                  <td><span className={r.side === "CE" ? "side long" : "side short"}>{r.side === "CE" ? "Call" : "Put"}</span></td>
                  <td className="hint">{r.ce_iv.toFixed(2)} / {r.pe_iv.toFixed(2)}</td>
                  <td>{num(r.short_strike, 0)} / {num(r.long_strike, 0)}</td>
                  <td>{num(r.credit, 1)} <span className="hint">(sell {num(r.sell_premium, 1)} / buy {num(r.buy_premium, 1)})</span></td>
                  <td className="move-up">{num(r.max_profit, 1)}</td>
                  <td>{num(r.max_risk, 1)} <span className="hint">(width {num(Math.abs(r.long_strike - r.short_strike), 1)} / credit {num(r.credit, 1)})</span></td>
                  <td>{r.lot_size ?? "—"}</td>
                  <td className="move-up">{r.max_profit_per_lot != null ? `₹${Math.round(r.max_profit_per_lot).toLocaleString("en-IN")}` : "—"}</td>
                  <td>{r.max_risk_per_lot != null ? `₹${Math.round(r.max_risk_per_lot).toLocaleString("en-IN")}` : "—"}</td>
                  <td><strong>{r.ror_pct.toFixed(0)}%</strong></td>
                  <td className="hint">{num(r.breakeven, 0)}</td>
                </tr>
              ))}
              {!daily.length && <tr><td colSpan={15} className="empty-cell">No candidate clears the 150% ret/risk bar today</td></tr>}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel cockpit" style={{ marginTop: 14 }}>
        <div className="panel-title">
          <h2>Signal history — returns per fired signal</h2>
          <div className="panel-title-controls">
            <label className="chk"><Coins size={15} />
              <select value={filterSym} onChange={(e) => setFilterSym(e.target.value)} style={{ minWidth: 160 }}>
                <option value="">All stocks</option>
                {symbols.map((s) => <option key={s.symbol} value={s.symbol}>{s.symbol} ({GROUP_LABEL[s.group] ?? s.group})</option>)}
              </select>
            </label>
            <MonthFilter months={months} value={filterMonth} onChange={setFilterMonth} />
          </div>
        </div>
        {hs ? (
          <div className="sell-bt">{hs.n} signals · win rate <b>{(hs.win_rate * 100).toFixed(0)}%</b> ·
            mean <b>{hs.ev_ror_pct >= 0 ? "+" : ""}{hs.ev_ror_pct}%</b> / median <b>{hs.median_ror_pct >= 0 ? "+" : ""}{hs.median_ror_pct}%</b> return-on-risk ·
            worst trade <b>{hs.worst_ror_pct}%</b> · worst drawdown <b>{hs.worst_dd_pct}%</b> of risk
            {!filterMonth && histRowsAll.length > histRows.length ? <span className="hint"> · showing the most recent {histRows.length} of {histRowsAll.length} — pick a month to see more</span> : null}</div>
        ) : null}
        <div className="table-wrap">
          <table className="freeze-cols">
            <thead><tr>
              <SortTh className="fz1" label="Signal date" sortKey="signal_date" sort={histSort.sort} onSort={histSort.onSort} />
              {allSyms ? <SortTh className="fz2" label="Symbol" sortKey="symbol" sort={histSort.sort} onSort={histSort.onSort} /> : null}
              <SortTh label="Exp / DTE" sortKey="dte" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh label="IV" sortKey="iv_ratio" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh label="Sell" sortKey="side" sort={histSort.sort} onSort={histSort.onSort} />
              <th>Short / Long</th>
              <SortTh label="Credit" sortKey="credit" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh label="Max profit" sortKey="max_profit" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh label="Max risk" sortKey="max_risk" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh label="Lot size" sortKey="lot_size" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh label="Max risk/lot" sortKey="max_risk_per_lot" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh label="Exit value" sortKey="exit_value" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh label="PnL/lot" sortKey="pnl_per_lot" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh label="Ret/risk" sortKey="ror_pct" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh label="Max DD" sortKey="max_dd_pct" sort={histSort.sort} onSort={histSort.onSort} />
              <th></th>
            </tr></thead>
            <tbody>
              {histSort.sorted.map((r, i) => (
                <tr key={`${r.symbol}-${r.signal_date}-${i}`}>
                  <td className="fz1">{r.signal_date}</td>
                  {allSyms ? <td className="fz2"><SymbolLink symbol={r.symbol} onOpen={setOpenSymbol} /> <span className="hint">{r.group.slice(0, 1)}</span></td> : null}
                  <td>{r.expiry.slice(5)} · {r.dte}d</td>
                  <td>{r.iv_ratio != null ? `${r.iv_ratio.toFixed(2)}×` : "—"}</td>
                  <td><span className={r.side === "CE" ? "side long" : "side short"}>{r.side === "CE" ? "Call" : "Put"}</span></td>
                  <td>{num(r.short_strike, 0)} / {num(r.long_strike, 0)}</td>
                  <td>{num(r.credit, 1)} <span className="hint">(sell {num(r.sell_premium, 1)} / buy {num(r.buy_premium, 1)})</span></td>
                  <td className="move-up">{num(r.max_profit, 1)} <span className="hint">{r.max_profit_per_lot != null ? `(₹${Math.round(r.max_profit_per_lot).toLocaleString("en-IN")}/lot)` : ""}</span></td>
                  <td>{num(r.max_risk, 1)} <span className="hint">(width {num(Math.abs(r.long_strike - r.short_strike), 1)} / credit {num(r.credit, 1)})</span></td>
                  <td>{r.lot_size ?? "—"}</td>
                  <td>{r.max_risk_per_lot != null ? `₹${Math.round(r.max_risk_per_lot).toLocaleString("en-IN")}` : "—"}</td>
                  <td>{num(r.exit_value, 1)}</td>
                  <td className={r.pnl == null ? "hint" : r.pnl >= 0 ? "move-up" : "move-down"}>
                    {r.pnl_per_lot != null
                      ? <>{r.pnl_per_lot >= 0 ? "+" : ""}₹{Math.round(r.pnl_per_lot).toLocaleString("en-IN")}
                          <span className="hint"> ({r.pnl != null && r.pnl >= 0 ? "+" : ""}{num(r.pnl, 1)}/share)</span></>
                      : r.pnl == null ? "…" : <>{r.pnl >= 0 ? "+" : ""}{num(r.pnl, 1)}</>}
                  </td>
                  <td className={r.ror_pct == null ? "hint" : r.ror_pct >= 0 ? "move-up" : "move-down"}>{r.ror_pct == null ? "…" : <>{r.ror_pct >= 0 ? "+" : ""}{r.ror_pct.toFixed(0)}%</>}</td>
                  <td className="move-down">{r.max_dd_pct.toFixed(0)}%</td>
                  <td>{r.outcome === "win" ? "✓" : r.outcome === "pending" ? <span className="hint" title="forward window not complete yet">…</span> : "✕"}</td>
                </tr>
              ))}
              {!histRows.length && <tr><td colSpan={allSyms ? 17 : 16} className="empty-cell">No signals</td></tr>}
            </tbody>
          </table>
        </div>
      </section>
      {openSymbol && <ChartModal symbol={openSymbol} onClose={() => setOpenSymbol(null)} />}
    </>
  );
}

// ----------------------------------------------------------------- Cash Signals (support/resistance zones)
type CashPick = {
  symbol: string; as_of: string; close: number | null; vol_ratio_20d: number | null; compression_composite: number | null;
  resistance_level?: number | null; support_level?: number | null; pct_beyond_level?: number | null;
  resistance_valid_touches?: number; resistance_zone_strength?: number; resistance_zone_age_weeks?: number | null;
  support_valid_touches?: number; support_zone_strength?: number; support_zone_age_weeks?: number | null;
  zone_box_width_pct?: number | null; weeks_in_box?: number | null;
};
type CashSignalsResp = {
  as_of: string | null; universe_size: number | null;
  breakout: CashPick[]; breakdown: CashPick[]; consolidation: CashPick[];
};
type CashSignalType = "breakout" | "breakdown" | "consolidation";
type CashHistRow = {
  signal_type: CashSignalType; symbol: string; signal_date: string; entry_close: number;
  resistance_level?: number | null; support_level?: number | null;
  resistance_zone_strength?: number | null; support_zone_strength?: number | null;
  fwd_return_5d?: number | null; fwd_return_10d?: number | null; fwd_return_20d?: number | null; held_10d?: boolean | null;
  zone_box_width_pct?: number | null; realized_range_10d_pct?: number | null;
  outcome: "win" | "loss";
};
type CashSummary = {
  n: number; win_rate: number;
  avg_fwd_return_10d?: number | null; median_fwd_return_10d?: number | null; held_rate_10d?: number | null;
  avg_realized_range_10d?: number | null;
} | null;
type CashHist = { rows: CashHistRow[]; summary: CashSummary };

function CashTrackRecord({ s }: { s: CashSummary }) {
  if (!s) return <span className="hint">no historical signals yet</span>;
  return (
    <>{s.n} historical signals · win rate <b>{(s.win_rate * 100).toFixed(0)}%</b>
      {s.avg_fwd_return_10d != null ? <> · mean fwd 10d <b>{s.avg_fwd_return_10d >= 0 ? "+" : ""}{s.avg_fwd_return_10d}%</b></> : null}
      {s.held_rate_10d != null ? <> · level held <b>{(s.held_rate_10d * 100).toFixed(0)}%</b> of the time</> : null}
      {s.avg_realized_range_10d != null ? <> · avg realized range <b>{s.avg_realized_range_10d}%</b></> : null}</>
  );
}

function CashZoneTable({ rows, side, onOpen }: { rows: CashPick[]; side: "breakout" | "breakdown"; onOpen: (s: string) => void }) {
  const levelKey = (side === "breakout" ? "resistance_level" : "support_level") as keyof CashPick;
  const touchesKey = (side === "breakout" ? "resistance_valid_touches" : "support_valid_touches") as keyof CashPick;
  const strengthKey = (side === "breakout" ? "resistance_zone_strength" : "support_zone_strength") as keyof CashPick;
  const ageKey = (side === "breakout" ? "resistance_zone_age_weeks" : "support_zone_age_weeks") as keyof CashPick;
  const s = useSortedRows(rows, strengthKey, "desc");
  return (
    <div className="table-wrap">
      <table className="freeze-cols">
        <thead><tr>
          <th className="fz1"></th>
          <SortTh className="fz2" label="Symbol" sortKey="symbol" sort={s.sort} onSort={s.onSort} />
          <SortTh label="Close" sortKey="close" sort={s.sort} onSort={s.onSort} />
          <SortTh label={side === "breakout" ? "Resistance" : "Support"} sortKey={levelKey} sort={s.sort} onSort={s.onSort} />
          <SortTh label="% beyond" sortKey="pct_beyond_level" sort={s.sort} onSort={s.onSort} />
          <SortTh label="Touches" sortKey={touchesKey} sort={s.sort} onSort={s.onSort} />
          <SortTh label="Zone strength" sortKey={strengthKey} sort={s.sort} onSort={s.onSort} />
          <SortTh label="Zone age" sortKey={ageKey} sort={s.sort} onSort={s.onSort} />
          <SortTh label="Volume vs 20d" sortKey="vol_ratio_20d" sort={s.sort} onSort={s.onSort} />
        </tr></thead>
        <tbody>
          {s.sorted.map((r, i) => (
            <tr key={r.symbol} className={i === 0 ? "sell-live" : "sell-secondary"}>
              <td className="fz1"><span className={`rank-badge ${i === 0 ? "primary" : ""}`}>#{i + 1}</span></td>
              <td className="fz2"><SymbolLink symbol={r.symbol} onOpen={onOpen} /></td>
              <td>{num(r.close ?? undefined, 1)}</td>
              <td>{num((r as unknown as Record<string, number | null>)[levelKey] ?? undefined, 1)}</td>
              <td className={side === "breakout" ? "move-up" : "move-down"}>
                {r.pct_beyond_level != null ? `${r.pct_beyond_level >= 0 ? "+" : ""}${r.pct_beyond_level.toFixed(2)}%` : "—"}</td>
              <td>{(r as unknown as Record<string, number | undefined>)[touchesKey] ?? "—"}</td>
              <td>{num((r as unknown as Record<string, number | undefined>)[strengthKey] ?? undefined, 2)}</td>
              <td>{(r as unknown as Record<string, number | null | undefined>)[ageKey] != null
                ? `${num((r as unknown as Record<string, number>)[ageKey], 0)}w` : "—"}</td>
              <td>{r.vol_ratio_20d != null ? `${r.vol_ratio_20d.toFixed(2)}×` : "—"}</td>
            </tr>
          ))}
          {!rows.length && <tr><td colSpan={8} className="empty-cell">No {side} today</td></tr>}
        </tbody>
      </table>
    </div>
  );
}

function CashConsolidationTable({ rows, onOpen }: { rows: CashPick[]; onOpen: (s: string) => void }) {
  const s = useSortedRows(rows, "zone_box_width_pct" as never, "asc");
  return (
    <div className="table-wrap">
      <table className="freeze-cols">
        <thead><tr>
          <th className="fz1"></th>
          <SortTh className="fz2" label="Symbol" sortKey="symbol" sort={s.sort} onSort={s.onSort} />
          <SortTh label="Close" sortKey="close" sort={s.sort} onSort={s.onSort} />
          <SortTh label="Support" sortKey="support_level" sort={s.sort} onSort={s.onSort} />
          <SortTh label="Resistance" sortKey="resistance_level" sort={s.sort} onSort={s.onSort} />
          <SortTh label="Box width" sortKey="zone_box_width_pct" sort={s.sort} onSort={s.onSort} />
          <SortTh label="Weeks in box" sortKey="weeks_in_box" sort={s.sort} onSort={s.onSort} />
        </tr></thead>
        <tbody>
          {s.sorted.map((r, i) => (
            <tr key={r.symbol} className={i === 0 ? "sell-live" : "sell-secondary"}>
              <td className="fz1"><span className={`rank-badge ${i === 0 ? "primary" : ""}`}>#{i + 1}</span></td>
              <td className="fz2"><SymbolLink symbol={r.symbol} onOpen={onOpen} /></td>
              <td>{num(r.close ?? undefined, 1)}</td>
              <td>{num(r.support_level ?? undefined, 1)}</td>
              <td>{num(r.resistance_level ?? undefined, 1)}</td>
              <td>{r.zone_box_width_pct != null ? `${r.zone_box_width_pct.toFixed(2)}%` : "—"}</td>
              <td>{r.weeks_in_box ?? "—"}</td>
            </tr>
          ))}
          {!rows.length && <tr><td colSpan={7} className="empty-cell">No tight consolidation today</td></tr>}
        </tbody>
      </table>
    </div>
  );
}

function CashSignals() {
  const [data, setData] = useState<CashSignalsResp | null>(null);
  const [symbols, setSymbols] = useState<{ symbol: string; group: string }[]>([]);
  const [catSummaries, setCatSummaries] = useState<Partial<Record<CashSignalType, CashSummary>>>({});
  const [signalType, setSignalType] = useState<CashSignalType | "">("");
  const [filterSym, setFilterSym] = useState("");
  const [filterMonth, setFilterMonth] = useState("");
  const [hist, setHist] = useState<CashHist | null>(null);
  const [openSymbol, setOpenSymbol] = useState<string | null>(null);

  useEffect(() => { getJson<CashSignalsResp>("/prod2/cash_signals").then(setData).catch(() => setData(null)); }, []);
  useEffect(() => { getJson<{ symbol: string; group: string }[]>("/prod2/symbols").then(setSymbols).catch(() => {}); }, []);
  useEffect(() => {
    (["breakout", "breakdown", "consolidation"] as const).forEach((t) => {
      getJson<CashHist>(`/prod2/cash_signal_history?signal_type=${t}`)
        .then((d) => setCatSummaries((prev) => ({ ...prev, [t]: d.summary })))
        .catch(() => {});
    });
  }, []);
  useEffect(() => {
    const q = new URLSearchParams({ ...(signalType ? { signal_type: signalType } : {}), ...(filterSym ? { symbol: filterSym } : {}) });
    getJson<CashHist>(`/prod2/cash_signal_history?${q}`).then(setHist).catch(() => setHist(null));
  }, [signalType, filterSym]);
  useEffect(() => setFilterMonth(""), [filterSym, signalType]);

  const months = useMemo(() => monthsOf(hist?.rows ?? [], (r) => r.signal_date), [hist]);
  const histRowsAll = useMemo(() => {
    const all = hist?.rows ?? [];
    return filterMonth ? all.filter((r) => r.signal_date.startsWith(filterMonth)) : all;
  }, [hist, filterMonth]);
  const histRows = filterMonth ? histRowsAll : histRowsAll.slice(0, MAX_UNFILTERED_HIST_ROWS);
  const hs = hist?.summary ?? null;
  const allSyms = filterSym === "";
  const histSort = useSortedRows(histRows, "signal_date" as never, "desc");

  return (
    <>
      <section className="panel cockpit">
        <div className="panel-title"><h2>Cash Signals — support / resistance zones</h2>
          <span>as of {data?.as_of ?? "—"} · {data?.universe_size ?? "—"} F&O symbols scanned</span></div>
        <div className="sell-explain">
          <div className="sell-rule">
            <strong>Zones</strong> Weekly swing highs/lows are clustered into support &amp; resistance bands (±2%); a touch only counts once price has
            retreated ≥5% since the last one, so sitting at a level for days doesn't inflate its strength. <b>Zone strength</b> = valid touches × log(zone age).
          </div>
          <div className="sell-rule">
            <strong>Breakout / breakdown</strong> Close has moved beyond the nearest zone's band within the last 5 trading days — ranked by the
            strength of the level that broke (a level tested many times over a long period is more significant than a fresh one).
          </div>
          <div className="sell-rule">
            <strong>Consolidation</strong> Price has sat inside a support/resistance box for ≥5 days without breaking out — ranked by tightest box width first.
            This is a "stays range-bound" read, not a directional call.
          </div>
        </div>

        <h3 style={{ margin: "14px 16px 4px" }}>Breaking out of resistance</h3>
        <div className="sell-bt" style={{ margin: "0 16px 8px" }}><CashTrackRecord s={catSummaries.breakout ?? null} /></div>
        <CashZoneTable rows={data?.breakout ?? []} side="breakout" onOpen={setOpenSymbol} />

        <h3 style={{ margin: "14px 16px 4px" }}>Breaking down through support</h3>
        <div className="sell-bt" style={{ margin: "0 16px 8px" }}><CashTrackRecord s={catSummaries.breakdown ?? null} /></div>
        <CashZoneTable rows={data?.breakdown ?? []} side="breakdown" onOpen={setOpenSymbol} />

        <h3 style={{ margin: "14px 16px 4px" }}>Tight consolidation</h3>
        <div className="sell-bt" style={{ margin: "0 16px 8px" }}><CashTrackRecord s={catSummaries.consolidation ?? null} /></div>
        <CashConsolidationTable rows={data?.consolidation ?? []} onOpen={setOpenSymbol} />
      </section>

      <section className="panel cockpit" style={{ marginTop: 14 }}>
        <div className="panel-title">
          <h2>Signal history — realized outcome per fired signal</h2>
          <div className="panel-title-controls">
            <label className="chk"><Waves size={15} />
              <select value={signalType} onChange={(e) => setSignalType(e.target.value as CashSignalType | "")} style={{ minWidth: 150 }}>
                <option value="">All types</option>
                <option value="breakout">Breakout</option>
                <option value="breakdown">Breakdown</option>
                <option value="consolidation">Consolidation</option>
              </select>
            </label>
            <label className="chk"><Coins size={15} />
              <select value={filterSym} onChange={(e) => setFilterSym(e.target.value)} style={{ minWidth: 160 }}>
                <option value="">All stocks</option>
                {symbols.map((s) => <option key={s.symbol} value={s.symbol}>{s.symbol}</option>)}
              </select>
            </label>
            <MonthFilter months={months} value={filterMonth} onChange={setFilterMonth} />
          </div>
        </div>
        {hs ? (
          <div className="sell-bt"><CashTrackRecord s={hs} />
            {!filterMonth && histRowsAll.length > histRows.length ? <span className="hint"> · showing the most recent {histRows.length} of {histRowsAll.length} — pick a month to see more</span> : null}
          </div>
        ) : null}
        <div className="table-wrap">
          <table className="freeze-cols">
            <thead><tr>
              <SortTh className="fz1" label="Signal" sortKey="signal_type" sort={histSort.sort} onSort={histSort.onSort} />
              <SortTh className="fz2" label="Signal date" sortKey="signal_date" sort={histSort.sort} onSort={histSort.onSort} />
              {allSyms ? <SortTh className="fz3" label="Symbol" sortKey="symbol" sort={histSort.sort} onSort={histSort.onSort} /> : null}
              <SortTh label="Entry close" sortKey="entry_close" sort={histSort.sort} onSort={histSort.onSort} />
              <th>Level</th><th>Strength / box</th>
              <SortTh label="Outcome" sortKey="outcome" sort={histSort.sort} onSort={histSort.onSort} />
            </tr></thead>
            <tbody>
              {histSort.sorted.map((r, i) => (
                <tr key={`${r.symbol}-${r.signal_date}-${i}`}>
                  <td className="fz1"><span className={`cash-signal-tag ${r.signal_type}`}>{r.signal_type}</span></td>
                  <td className="fz2">{r.signal_date}</td>
                  {allSyms ? <td className="fz3"><SymbolLink symbol={r.symbol} onOpen={setOpenSymbol} /></td> : null}
                  <td>{num(r.entry_close, 1)}</td>
                  <td>{num((r.signal_type === "breakdown" ? r.support_level : r.resistance_level) ?? undefined, 1)}</td>
                  <td>{r.signal_type === "consolidation"
                    ? (r.zone_box_width_pct != null ? `${r.zone_box_width_pct.toFixed(2)}%` : "—")
                    : num((r.signal_type === "breakdown" ? r.support_zone_strength : r.resistance_zone_strength) ?? undefined, 2)}</td>
                  <td className={r.outcome === "win" ? "move-up" : "move-down"}>
                    {r.outcome === "win" ? "✓" : "✕"}
                    {r.fwd_return_10d != null ? <span className="hint"> ({r.fwd_return_10d >= 0 ? "+" : ""}{r.fwd_return_10d}%)</span> : null}
                    {r.realized_range_10d_pct != null ? <span className="hint"> (range {r.realized_range_10d_pct}%)</span> : null}
                  </td>
                </tr>
              ))}
              {!histRows.length && <tr><td colSpan={allSyms ? 7 : 6} className="empty-cell">No signals</td></tr>}
            </tbody>
          </table>
        </div>
      </section>
      {openSymbol && <ChartModal symbol={openSymbol} onClose={() => setOpenSymbol(null)} />}
    </>
  );
}

// ----------------------------------------------------------------- Indices (NIFTY intraday breakout)
type IntradayBar = { ts: string; open: number; high: number; low: number; close: number };
// Spot-only (predicted vs realized move %), not a straddle-premium P&L trade -- the underlying
// strategy is the regime-switching sell-spread/naked-buy off the live chain (see
// production/nifty_live_poller.py); this replay just checks the model+exit-state-machine
// against history, not a specific option structure's fill.
type IntradayTrade = {
  date: string; entry_ts: string; exit_ts: string | null;
  entry_underlying: number; exit_underlying: number | null;
  predicted_move_pct: number; realized_move_pct: number | null;
  direction: "up" | "down" | null; exit_reason: string | null;
};
type IntradaySignalsResp = {
  as_of: string | null; horizon_bars: number | null; move_threshold_pct: number | null; train_cutoff: string | null; exit_rule: string | null;
  backtest_summary: { n: number; n_resolved: number; hit_rate: number; mean_abs_realized_move_pct: number; exit_reasons: Record<string, number> } | null;
  trades: IntradayTrade[];
};
type DailyBar = { date: string; open: number; high: number; low: number; close: number };
type NiftyTF = "5m" | "15m" | "1D" | "1W";
const NIFTY_TF_LABEL: Record<NiftyTF, string> = { "5m": "5 min", "15m": "15 min", "1D": "1 day", "1W": "1 week" };

// Groups consecutive intraday bars into `n`-bar buckets (e.g. 5m -> 15m is n=3). Session bars
// start at 9:15 on a 5-min grid, so grouping-by-3 lands on clean 9:15/9:30/... 15-min boundaries
// with no explicit alignment needed.
function resampleBars(bars: IntradayBar[], n: number): IntradayBar[] {
  if (n <= 1) return bars;
  const out: IntradayBar[] = [];
  for (let i = 0; i < bars.length; i += n) {
    const chunk = bars.slice(i, i + n);
    out.push({
      ts: chunk[0].ts, open: chunk[0].open, close: chunk[chunk.length - 1].close,
      high: Math.max(...chunk.map((b) => b.high)), low: Math.min(...chunk.map((b) => b.low)),
    });
  }
  return out;
}
// IntradayBar (ts/OHLC) -> PricePoint shape so the stock chart's srLevels()/computeAutoDD() can
// be reused as-is; volume/delivQty/picked are unused by those two functions.
function barToPricePoint(b: IntradayBar): PricePoint {
  return { date: b.ts, open: b.open, high: b.high, low: b.low, close: b.close, volume: 0, delivQty: 0, picked: false };
}
// Binary search: smallest index i with bars[i].ts >= ts (bars sorted ascending); bars.length if none.
function idxTsOnOrAfter(bars: IntradayBar[], ts: string): number {
  let lo = 0, hi = bars.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (bars[mid].ts < ts) lo = mid + 1; else hi = mid; }
  return lo;
}

// Native 5-min NIFTY candles for ONE trading day, with entry/exit markers for that day's
// fired trade(s). Forks Candles' SVG/scale/zoom-pan/tooltip skeleton (same hand-rolled-SVG
// convention, no charting library) rather than overloading Candles with a new timeframe --
// Candles' resample/srLevels/tick-label logic is date-string/day-granularity-specific and
// a single trading day (~75 bars) doesn't need that machinery (no weekly S/R, no volume --
// the raw index has none). Markers are new territory: `picked` elsewhere in this app is only
// ever a *table* annotation, never drawn on a chart.
function IntradayCandles({ series, trades, tf, dailyFull, weeklyFull, intraZoneSource, moveThresholdPct }: {
  series: IntradayBar[]; trades: IntradayTrade[]; tf: NiftyTF;
  dailyFull: IntradayBar[]; weeklyFull: IntradayBar[]; intraZoneSource: IntradayBar[];
  moveThresholdPct: number;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const DEFAULT_TRAILING = 150;
  const defWin = (len: number) => ({ s: Math.max(0, len - DEFAULT_TRAILING), e: len });
  const [win, setWin] = useState<{ s: number; e: number }>(defWin(series.length));
  const [sel, setSel] = useState<number | null>(null);
  // Vertical compress/stretch (TradingView-style): 1 = auto-fit the visible price range, >1
  // zooms in (stretches -- candles look taller), <1 zooms out (compresses -- candles look
  // flatter). Independent of horizontal zoom/pan (`win`). See the price-axis drag/wheel/
  // double-click handlers below.
  const [vZoom, setVZoom] = useState(1);
  // The 60s live-data poll (IndicesSignals) re-fetches `series` with a brand-new array identity
  // every time, even when it's the SAME trailing window with just one more candle appended --
  // resetting win/sel on every `series` change made the chart visibly snap back to the default
  // view every minute, reading as "the page refreshed" even though nothing else did. Only reset
  // on a genuinely NEW context (date/timeframe switch, detected via the window's start boundary
  // moving or the series shrinking); a same-context growth (live poll) instead silently slides
  // the window forward by the same delta ONLY if it was already pinned to "show latest" --
  // if the user had scrolled back into history, their position is left untouched.
  const prevSeriesRef = useRef<{ len: number; firstTs: string | null }>({ len: 0, firstTs: null });
  useEffect(() => {
    const prev = prevSeriesRef.current;
    const firstTs = series.length ? series[0].ts : null;
    const isNewContext = firstTs !== prev.firstTs || series.length < prev.len;
    if (isNewContext) {
      setWin(defWin(series.length));
      setSel(null);
      setVZoom(1);
    } else if (series.length > prev.len) {
      const delta = series.length - prev.len;
      setWin((w) => (w.e >= prev.len ? { s: w.s + delta, e: w.e + delta } : w));
    }
    prevSeriesRef.current = { len: series.length, firstTs };
  }, [series]);
  const intraday = tf === "5m" || tf === "15m";
  const showIntraZones = intraday;
  // On a multi-day 5m/15m chart, the session-open (09:15) bar shows its DATE instead of the time
  // -- otherwise every day boundary just reads "09:15" with no way to tell which day it is.
  const tickLabel = (ts: string) => {
    if (!intraday) return ts.slice(5, 10);
    return ts.slice(11, 16) === "09:15" ? ts.slice(5, 10) : ts.slice(11, 16);
  };
  // A tick is a DATE (vs. a time) only on the intraday 5m/15m views, at the 09:15 day-boundary
  // bar -- colored blue so it visually stands out from the surrounding HH:MM time ticks.
  const isDateTick = (ts: string) => intraday && ts.slice(11, 16) === "09:15";
  const dateColor = "#2563b0";

  const W = 900, H = 340, padX = 40, padTop = 14, padBot = 34;
  const total = series.length;
  const s = Math.max(0, Math.min(win.s, Math.max(0, total - 2)));
  const e = Math.max(s + 2, Math.min(win.e, total));
  const vis = series.slice(s, e);
  const n = vis.length;
  const slot = n > 0 ? (W - 2 * padX) / n : 0;
  const cw = Math.max(1, Math.min(10, slot * 0.7));
  const x = (i: number) => padX + slot * (i + 0.5);
  const idxOf = (ts: string) => series.findIndex((b) => b.ts === ts);

  // Three S/R zone layers, each scoped to never look past the last visible bar's timestamp (no
  // lookahead -- same discipline as Candles' weekly zones for stocks). Weekly/daily are solid,
  // shown on every timeframe; the tight intraday layer (15-min bars, 0.10% cluster tolerance
  // instead of stocks'/daily's/weekly's 2% -- NIFTY moves far less in absolute % terms within a
  // session) is dotted and shown only on the 5m/15m views.
  const lastVisibleTs = vis.length ? vis[vis.length - 1].ts : null;
  const weeklyLevels = useMemo(() => {
    if (!lastVisibleTs || weeklyFull.length < 5) return [];
    const scoped = weeklyFull.filter((b) => b.ts <= lastVisibleTs);
    if (scoped.length < 5) return [];
    const pts = scoped.map(barToPricePoint);
    const dd = computeAutoDD(pts) ?? 0.15;
    let ath = -Infinity;
    for (const p of pts) if (p.high > ath) ath = p.high;
    return srLevels(pts, dd, ath);
  }, [weeklyFull, lastVisibleTs]);
  const dailyLevels = useMemo(() => {
    if (!lastVisibleTs || dailyFull.length < 5) return [];
    const scoped = dailyFull.filter((b) => b.ts <= lastVisibleTs);
    if (scoped.length < 5) return [];
    const pts = scoped.map(barToPricePoint);
    const dd = computeAutoDD(pts) ?? 0.10;
    let ath = -Infinity;
    for (const p of pts) if (p.high > ath) ath = p.high;
    return srLevels(pts, dd, ath);
  }, [dailyFull, lastVisibleTs]);
  const intraZoneLevels = useMemo(() => {
    if (!showIntraZones || !lastVisibleTs || intraZoneSource.length < 5) return [];
    let scoped = intraZoneSource.filter((b) => b.ts <= lastVisibleTs);
    if (scoped.length < 5) return [];
    // Trailing 9 TRADING days only (2026-08, explicit user decision): this layer is meant to
    // capture RECENT intraday price structure, not the full ~30-trading-day window the chart
    // loads for panning/scrollback -- without this bound, older bars diluted it with stale
    // levels far outside what's actually relevant to a 5m/15m trader right now.
    const uniqDates = Array.from(new Set(scoped.map((b) => b.ts.slice(0, 10)))).sort();
    const cutoffDate = uniqDates.length > 9 ? uniqDates[uniqDates.length - 9] : uniqDates[0];
    scoped = scoped.filter((b) => b.ts.slice(0, 10) >= cutoffDate);
    if (scoped.length < 5) return [];
    const pts = scoped.map(barToPricePoint);
    let ath = -Infinity;
    for (const p of pts) if (p.high > ath) ath = p.high;
    return srLevels(pts, 0.005, ath, 0.001);
  }, [intraZoneSource, lastVisibleTs, showIntraZones]);

  useEffect(() => {
    const node = wrapRef.current;
    if (!node) return;
    const onWheel = (ev: WheelEvent) => {
      ev.preventDefault();
      const rect = node.getBoundingClientRect();
      const frac = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
      // Scrolling over the LEFT price-axis margin (where the price grid labels live) zooms
      // vertically instead of the usual horizontal time-zoom -- same TradingView convention as
      // the drag handler below, just via the wheel.
      if (frac * W < padX) {
        setVZoom((z) => Math.max(0.2, Math.min(6, z * (ev.deltaY < 0 ? 1.1 : 1 / 1.1))));
        return;
      }
      const cursorIdx = s + frac * n;
      const factor = ev.deltaY < 0 ? 0.82 : 1.22;
      let width = Math.round(n * factor);
      width = Math.max(6, Math.min(total, width));
      let ns = Math.round(cursorIdx - frac * width);
      ns = Math.max(0, Math.min(total - width, ns));
      setWin({ s: ns, e: ns + width });
    };
    node.addEventListener("wheel", onWheel, { passive: false });
    return () => node.removeEventListener("wheel", onWheel);
  }, [s, n, total]);

  // Vertical drag-to-stretch/compress on the price axis (TradingView convention): mousedown in
  // the left margin starts a drag; moving the mouse up stretches (vZoom increases, candles get
  // taller), moving down compresses. Double-click the same margin resets to auto-fit.
  useEffect(() => {
    const node = wrapRef.current;
    if (!node) return;
    let dragging = false, startY = 0, startZoom = 1;
    const inAxis = (clientX: number) => {
      const rect = node.getBoundingClientRect();
      return ((clientX - rect.left) / rect.width) * W < padX;
    };
    const onDown = (ev: MouseEvent) => {
      if (!inAxis(ev.clientX)) return;
      dragging = true; startY = ev.clientY; startZoom = vZoom;
      ev.preventDefault();
    };
    const onMove = (ev: MouseEvent) => {
      if (!dragging) return;
      const deltaPx = startY - ev.clientY;   // up = positive = stretch
      setVZoom(Math.max(0.2, Math.min(6, startZoom * Math.exp(deltaPx / 150))));
    };
    const onUp = () => { dragging = false; };
    const onDbl = (ev: MouseEvent) => {
      if (!inAxis(ev.clientX)) return;
      ev.stopPropagation();   // don't also trigger the whole-chart double-click (pan/selection reset)
      setVZoom(1);
    };
    node.addEventListener("mousedown", onDown);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    node.addEventListener("dblclick", onDbl, true);   // capture phase, so it runs before the chart's own onDoubleClick
    return () => {
      node.removeEventListener("mousedown", onDown);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      node.removeEventListener("dblclick", onDbl, true);
    };
  }, [vZoom]);

  // keyboard: left/right arrows pan (10% of window per press), up/down zoom -- same convention
  // as Candles (stock charts). Attached to the chart's own wrapper node (not `window`), so arrow
  // keys only move THIS chart once it's been clicked into focus -- not any time an arrow key is
  // pressed anywhere on the page. `tabIndex` on the wrapper (below) is what makes it focusable.
  useEffect(() => {
    const node = wrapRef.current;
    if (!node) return;
    const onKey = (ev: KeyboardEvent) => {
      if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(ev.key)) return;
      ev.preventDefault();
      if (ev.key === "ArrowLeft" || ev.key === "ArrowRight") {
        const width = e - s;
        const step = Math.max(1, Math.round(width * 0.1));
        if (ev.key === "ArrowLeft") { const ns = Math.max(0, s - step); setWin({ s: ns, e: ns + width }); }
        else { const ne = Math.min(total, e + step); setWin({ s: ne - width, e: ne }); }
      } else {
        const factor = ev.key === "ArrowUp" ? 0.7 : 1.4;
        const w = Math.max(6, Math.min(total, Math.round(n * factor)));
        const mid = s + n / 2;
        const ns = Math.max(0, Math.min(total - w, Math.round(mid - w / 2)));
        setWin({ s: ns, e: ns + w });
      }
    };
    node.addEventListener("keydown", onKey);
    return () => node.removeEventListener("keydown", onKey);
  }, [s, e, n, total]);

  if (total < 2) return <span className="hint">No data</span>;

  // toolbar ‹/› buttons: pan by half the current window (a bigger "page" step than the arrow
  // keys' 10%, for quickly moving across days at a glance).
  const panBy = (dir: -1 | 1) => {
    const width = e - s;
    const step = Math.max(1, Math.round(width * 0.5));
    if (dir < 0) { const ns = Math.max(0, s - step); setWin({ s: ns, e: ns + width }); }
    else { const ne = Math.min(total, e + step); setWin({ s: ne - width, e: ne }); }
  };

  // Only trades whose entry OR exit marker actually falls in the visible window [s, e) should
  // stretch the y-axis -- `trades` can span the whole loaded multi-day range (up to 30 trading
  // days), and an off-screen trade from a very different price level was blowing the y-range out
  // to fit a marker nobody can even see (auto-zoom must track what's ON SCREEN, not what's loaded).
  const visTrades = trades.filter((t) => {
    const ei = idxOf(t.entry_ts), xi = t.exit_ts ? idxOf(t.exit_ts) : -1;
    return (ei >= s && ei < e) || (xi >= s && xi < e);
  });
  const rawHi = Math.max(...vis.map((p) => p.high), ...visTrades.map((t) => Math.max(t.entry_underlying, t.exit_underlying ?? t.entry_underlying)));
  const rawLo = Math.min(...vis.map((p) => p.low), ...visTrades.map((t) => Math.min(t.entry_underlying, t.exit_underlying ?? t.entry_underlying)));
  // vZoom=1 reproduces the old fixed-6%-padding auto-fit exactly; >1 shrinks the effective
  // range around the same midpoint (stretch -- candles get taller), <1 grows it (compress).
  const rawMid = (rawHi + rawLo) / 2;
  const half = ((rawHi - rawLo) / 2 || 1) * 1.06 / vZoom;
  const hi = rawMid + half, lo = rawMid - half;
  const y = (v: number) => padTop + (1 - (v - lo) / (hi - lo || 1)) * (H - padTop - padBot);
  const up = "#167c80", down = "#9a2431";
  const atDefault = s === Math.max(0, total - DEFAULT_TRAILING) && e === total && vZoom === 1;
  const grid = niceTicks(lo, hi, 8);

  const idxFromClientX = (clientX: number): number => {
    const rect = wrapRef.current!.getBoundingClientRect();
    const svgX = ((clientX - rect.left) / rect.width) * W;
    return Math.max(0, Math.min(n - 1, Math.round((svgX - padX) / slot - 0.5)));
  };

  const step = Math.max(1, Math.ceil(n / 10));
  const ticks: number[] = [];
  for (let i = 0; i < n; i += step) ticks.push(i);
  if (ticks[ticks.length - 1] !== n - 1) ticks.push(n - 1);
  // explicitly include every day-boundary (09:15) bar as its own tick, so a multi-day 5m/15m
  // chart always shows which day you're looking at, not just whichever evenly-spaced tick
  // happens to land near it.
  if (intraday) {
    for (let i = 0; i < n; i++) if (vis[i].ts.slice(11, 16) === "09:15" && !ticks.includes(i)) ticks.push(i);
    ticks.sort((a, b) => a - b);
  }

  const hp = sel != null && sel < n ? vis[sel] : null;
  const prevClose = hp != null ? series[s + sel! - 1]?.close : null;
  const chgPct = hp != null && prevClose ? ((hp.close - prevClose) / prevClose) * 100 : null;
  const tipLeftPct = sel != null ? (x(sel) / W) * 100 : 0;
  const tipRight = tipLeftPct > 62;

  return (
    <div ref={wrapRef} className="candles-wrap" style={{ position: "relative" }} tabIndex={0}
         // Single click just dismisses any open popup (click-away-to-close); the candle-detail
         // popup itself now only opens on DOUBLE click (2026-08, explicit user decision -- it
         // was popping up on every single click, too eager while panning/scrubbing the chart).
         // Zoom-reset moved off double-click since the "Reset" button below already covers it
         // and double-click's job here is now the popup, not zoom.
         onClick={() => setSel(null)}
         onDoubleClick={(ev) => { const idx = idxFromClientX(ev.clientX); setSel((prev) => (prev === idx ? null : idx)); }}>
      <div className="candles-toolbar">
        <button type="button" title="Pan back" disabled={s <= 0} onClick={() => panBy(-1)}>‹</button>
        <button type="button" title="Pan forward" disabled={e >= total} onClick={() => panBy(1)}>›</button>
        <button type="button" title="Reset zoom" disabled={atDefault} onClick={() => { setWin(defWin(total)); setVZoom(1); }}>Reset</button>
      </div>
      {/* Purely a cursor hint over the draggable price-axis strip (left margin) -- the actual
          drag/wheel/double-click handlers are attached at the wrapper level above, not here. */}
      <div title="Drag to stretch/compress vertically · double-click to reset"
           style={{ position: "absolute", left: 0, top: `${(padTop / H) * 100}%`, bottom: `${(padBot / H) * 100}%`,
                    width: `${(padX / W) * 100}%`, cursor: "ns-resize" }} />
      <svg viewBox={`0 0 ${W} ${H}`} className="spark">
        {grid.map((g, gi) => (
          <g key={`grid${gi}`}>
            <line x1={padX} x2={W - padX} y1={y(g)} y2={y(g)} stroke="#e7ecea" strokeWidth={1} />
            <text x={padX - 6} y={y(g) + 3} fontSize={10} fill="#8a978f" textAnchor="end">{num(g, 0)}</text>
          </g>
        ))}
        {hp != null ? <line x1={x(sel!)} x2={x(sel!)} y1={padTop} y2={H - padBot} stroke="#b9c6c0" strokeWidth={1} strokeDasharray="3 3" /> : null}
        {vis.map((p, i) => {
          const rising = p.close >= p.open;
          const color = rising ? up : down;
          const yo = y(p.open), yc = y(p.close);
          const top = Math.min(yo, yc), bh = Math.max(1, Math.abs(yc - yo));
          return (
            <g key={s + i}>
              <line x1={x(i)} x2={x(i)} y1={y(p.high)} y2={y(p.low)} stroke={color} strokeWidth={1} />
              <rect x={x(i) - cw / 2} y={top} width={cw} height={bh} fill={color} />
            </g>
          );
        })}
        {/* entry/exit trade markers: shaded band from entry to exit + ▲/▼ glyphs, colored by
            whether the realized move hit the predicted-move threshold (not premium P&L --
            this replay checks the model+exit state machine, not a specific option fill) */}
        {trades.map((t, ti) => {
          const ei = idxOf(t.entry_ts) - s;
          if (ei < 0 || ei >= n) return null;
          const hasExit = t.exit_ts != null && t.exit_underlying != null;
          const xi = hasExit ? idxOf(t.exit_ts!) - s : -1;
          const hit = t.realized_move_pct != null && Math.abs(t.realized_move_pct) >= moveThresholdPct;
          const color = hit ? up : down;
          const xiVis = xi >= 0 && xi < n;
          return (
            <g key={`trade${ti}`}>
              {xiVis ? <rect x={x(ei)} y={padTop} width={Math.max(0, x(xi) - x(ei))} height={H - padTop - padBot}
                             fill={color} opacity={0.07} /> : null}
              <polygon points={`${x(ei)},${y(t.entry_underlying) - 9} ${x(ei) - 5},${y(t.entry_underlying) - 1} ${x(ei) + 5},${y(t.entry_underlying) - 1}`} fill={up} />
              <text x={x(ei)} y={y(t.entry_underlying) - 12} fontSize={9} fill={up} textAnchor="middle">Entry</text>
              {xiVis && t.exit_underlying != null ? (
                <>
                  <polygon points={`${x(xi)},${y(t.exit_underlying) + 9} ${x(xi) - 5},${y(t.exit_underlying) + 1} ${x(xi) + 5},${y(t.exit_underlying) + 1}`} fill={color} />
                  <text x={x(xi)} y={y(t.exit_underlying) + 20} fontSize={9} fill={color} textAnchor="middle">
                    {t.realized_move_pct != null ? `${t.realized_move_pct >= 0 ? "+" : ""}${t.realized_move_pct.toFixed(2)}%` : "—"}
                  </text>
                </>
              ) : null}
            </g>
          );
        })}
        {/* weekly-based S/R zones (solid) -- always shown, on every timeframe */}
        {weeklyLevels.filter((L) => L.price >= lo && L.price <= hi).map((L, li) => {
          const color = L.kind === "ATH" ? "#2563b0" : "#5c4a9e";
          const relIdx = idxTsOnOrAfter(series, L.firstDate) - s;
          const xStart = Math.min(W - padX, Math.max(padX, x(relIdx)));
          return (
            <g key={`wlv${li}`}>
              <line x1={xStart} x2={W - padX} y1={y(L.price)} y2={y(L.price)} stroke={color} strokeWidth={0.55} />
              {/* label sits in the right margin PAST the line's end (padX-wide band the candles
                  never draw into), not overlapping-left over the line/candles the way a
                  textAnchor="end" label anchored just inside the plot edge used to (2026-08,
                  explicit user decision -- was landing on top of the most recent candles) */}
              <text x={W - padX + 3} y={y(L.price) + 2} fontSize={7} fill={color} textAnchor="start">{num(L.price, 1)}×{L.touches}(W)</text>
            </g>
          );
        })}
        {/* daily-based S/R zones (solid) -- always shown, on every timeframe */}
        {dailyLevels.filter((L) => L.price >= lo && L.price <= hi).map((L, li) => {
          const color = L.kind === "ATH" ? "#2563b0" : "#33443d";
          const relIdx = idxTsOnOrAfter(series, L.firstDate) - s;
          const xStart = Math.min(W - padX, Math.max(padX, x(relIdx)));
          return (
            <g key={`dlv${li}`}>
              <line x1={xStart} x2={W - padX} y1={y(L.price)} y2={y(L.price)} stroke={color} strokeWidth={0.55} />
              <text x={W - padX + 3} y={y(L.price) + 11} fontSize={7} fill={color} textAnchor="start">{num(L.price, 1)}×{L.touches}(D)</text>
            </g>
          );
        })}
        {/* 15-min-based S/R zones (0.10% tolerance), dotted + distinct color -- only on 5m/15m
            views, where a tight intraday zone actually means something (see showIntraZones) */}
        {intraZoneLevels.filter((L) => L.price >= lo && L.price <= hi).map((L, li) => {
          const color = L.kind === "ATH" ? "#2563b0" : "#a3651f";
          const relIdx = idxTsOnOrAfter(series, L.firstDate) - s;
          const xStart = Math.min(W - padX, Math.max(padX, x(relIdx)));
          return (
            <g key={`ilv${li}`}>
              <line x1={xStart} x2={W - padX} y1={y(L.price)} y2={y(L.price)} stroke={color} strokeWidth={0.6} strokeDasharray="2 3" />
              <text x={W - padX + 3} y={y(L.price) + 20} fontSize={7} fill={color} textAnchor="start">{num(L.price, 1)}×{L.touches}(15m)</text>
            </g>
          );
        })}
        {ticks.map((i) => (
          <text key={i} x={x(i)} y={H - padBot + 16} fontSize={7} fill={isDateTick(vis[i].ts) ? dateColor : "#60706a"}
                fontWeight={isDateTick(vis[i].ts) ? 700 : 400}
                textAnchor={i === 0 ? "start" : i === n - 1 ? "end" : "middle"}>{tickLabel(vis[i].ts)}</text>
        ))}
      </svg>
      {hp ? (
        <div className="candle-tip" style={{ [tipRight ? "right" : "left"]: `calc(${tipRight ? 100 - tipLeftPct : tipLeftPct}% + 10px)`, top: 8 }}>
          <strong style={isDateTick(hp.ts) ? { color: dateColor } : undefined}>{tickLabel(hp.ts)}</strong>
          {chgPct != null ? (
            <span>Chg <b style={{ color: chgPct >= 0 ? up : down }}>{chgPct >= 0 ? "+" : ""}{num(chgPct, 2)}%</b></span>
          ) : null}
          <span>O <b>{num(hp.open, 1)}</b></span>
          <span>H <b>{num(hp.high, 1)}</b></span>
          <span>L <b>{num(hp.low, 1)}</b></span>
          <span>C <b style={{ color: hp.close >= hp.open ? up : down }}>{num(hp.close, 1)}</b></span>
        </div>
      ) : null}
    </div>
  );
}

// "YYYY-MM" -> "August 2026", for the date dropdown's month optgroups.
function monthLabel(ym: string): string {
  const [y, m] = ym.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, 1)).toLocaleString("en-US", { month: "long", year: "numeric", timeZone: "UTC" });
}

function IndicesSignals() {
  const [date, setDate] = useState<string>("");
  const [bars5mRange, setBars5mRange] = useState<IntradayBar[]>([]);
  const [dailyBars, setDailyBars] = useState<IntradayBar[]>([]);
  const [signals, setSignals] = useState<IntradaySignalsResp | null>(null);
  const [tf, setTf] = useState<NiftyTF>("5m");

  const fetchSignals = useCallback(() => {
    getJson<IntradaySignalsResp>("/prod2/nifty_intraday_signals").then(setSignals).catch(() => setSignals(null));
  }, []);
  useEffect(() => { fetchSignals(); }, [fetchSignals]);
  // NIFTY's own daily OHLC (years of history) -- feeds the 1D/1W timeframe views, the date
  // dropdown (ALL trading days, not just days a signal fired), and the daily/weekly S/R zone
  // layers shown on every timeframe. Fetched once, independent of `date`/`tf`.
  useEffect(() => {
    getJson<{ bars: DailyBar[] }>("/prod2/nifty_daily_bars?days=2500")
      .then((d) => setDailyBars(d.bars.map((b) => ({ ts: b.date, open: b.open, high: b.high, low: b.low, close: b.close }))))
      .catch(() => setDailyBars([]));
  }, []);
  // Default the date picker to TODAY's actual calendar date, not the latest daily bar -- the
  // daily-bars table only gets today's row after an EOD downloader run, so defaulting to its
  // last entry showed yesterday's chart even mid-session. /prod2/nifty_intraday_bars_range
  // gracefully falls back to the most recent trading day if today has no data yet (weekend,
  // holiday, or before the live poller's first poll), so defaulting to today is always safe.
  useEffect(() => { if (!date) setDate(new Date().toLocaleDateString("en-CA")); }, [date]);
  // 30 trailing trading days of 5-min bars ENDING at the selected date (never past it) -- enough
  // for a 150-trailing-candle 5m/15m view plus scrollback for the </>/arrow-key panning, without
  // clipping the chart to a single day the way the old per-date-only fetch did. When `date` is
  // today, the API splices in production/nifty_live_poller.py's live bars (09:15 onward).
  const fetchRange = useCallback(() => {
    if (!date) return;
    getJson<{ end_date: string; bars: IntradayBar[] }>(`/prod2/nifty_intraday_bars_range?end_date=${date}&trading_days=30`)
      .then((d) => setBars5mRange(d.bars)).catch(() => setBars5mRange([]));
  }, [date]);
  useEffect(() => { fetchRange(); }, [fetchRange]);
  // Push, not poll: a fixed-timer refetch was out of sync with the poller's actual 5-min cadence
  // (up to 60s stale) and independently was what caused the chart-reset bug fixed earlier --
  // subscribe to /prod2/nifty_live_events (SSE) instead, which the API server only pushes an
  // `update` on when production/nifty_live_poller.py has actually written a new poll cycle.
  // Refetches BOTH the chart bars and the signals table on every push, so a newly-fired signal
  // shows up immediately too, not just the candle. Only subscribed while viewing today (a past
  // date's chart is static, nothing there will ever change).
  useEffect(() => {
    if (!date || date !== new Date().toLocaleDateString("en-CA")) return;
    const es = new EventSource(`${API_BASE}/prod2/nifty_live_events`);
    es.addEventListener("update", () => { fetchRange(); fetchSignals(); });
    es.onerror = () => { /* EventSource auto-reconnects on its own; nothing to do here */ };
    return () => es.close();
  }, [date, fetchRange, fetchSignals]);

  const weeklyBars = useMemo(() => {
    if (!dailyBars.length) return [];
    const resampled = resample(dailyBars.map(barToPricePoint), "W");
    return resampled.map((p) => ({ ts: p.date, open: p.open, high: p.high, low: p.low, close: p.close }));
  }, [dailyBars]);
  const bars15mRange = useMemo(() => resampleBars(bars5mRange, 3), [bars5mRange]);

  // Date dropdown: ALL trading days in the trailing 12 months, grouped by month -- regardless of
  // whether a signal fired that day (previously sourced from signals.json's trade dates only,
  // which silently hid every day with no fired signal).
  const dateGroups = useMemo(() => {
    if (!dailyBars.length) return [] as [string, string[]][];
    const latest = dailyBars[dailyBars.length - 1].ts;
    const cutoff = new Date(latest + "T00:00:00Z"); cutoff.setUTCFullYear(cutoff.getUTCFullYear() - 1);
    const cutoffStr = cutoff.toISOString().slice(0, 10);
    const groups = new Map<string, string[]>();
    for (const b of dailyBars) {
      if (b.ts < cutoffStr) continue;
      const key = b.ts.slice(0, 7);
      (groups.get(key) ?? groups.set(key, []).get(key)!).push(b.ts);
    }
    // Today's row only lands in dailyBars after an EOD downloader run -- without it, `date`
    // (defaulted to today above) has no matching <option>, so the <select> silently falls back
    // to displaying whichever day IS first in the list (yesterday) even though the chart itself
    // is correctly showing today's live-spliced data. Add it explicitly so the dropdown's
    // displayed value always matches what's actually on screen.
    const today = new Date().toLocaleDateString("en-CA");
    const todayKey = today.slice(0, 7);
    const group = groups.get(todayKey);
    if (group) { if (!group.includes(today)) group.push(today); }
    else groups.set(todayKey, [today]);
    return [...groups.entries()].sort((a, b) => b[0].localeCompare(a[0])).map(([k, ds]) => [k, ds.slice().reverse()] as [string, string[]]);
  }, [dailyBars]);

  const isIntraday = tf === "5m" || tf === "15m";
  const chartBars = useMemo(() => {
    if (tf === "5m") return bars5mRange;
    if (tf === "15m") return bars15mRange;
    if (tf === "1D") return dailyBars;
    return weeklyBars;
  }, [tf, bars5mRange, bars15mRange, dailyBars, weeklyBars]);

  // Signals fired within the loaded intraday range (not just the single selected date) -- the
  // chart now spans multiple trailing days, so markers/table cover all of them. 5m-only: trade
  // timestamps are 5-min-precise and won't cleanly align with 15m-resampled bars.
  const rangeDates = useMemo(() => new Set(bars5mRange.map((b) => b.ts.slice(0, 10))), [bars5mRange]);
  const rangeTrades = useMemo(() => (tf === "5m" ? (signals?.trades ?? []).filter((t) => rangeDates.has(t.date)) : []), [signals, rangeDates, tf]);
  const bt = signals?.backtest_summary;

  return (
    <>
      <section className="panel cockpit">
        <div className="panel-title"><h2>Indices — NIFTY intraday regime (magnitude model)</h2>
          <span>horizon {signals?.horizon_bars ?? "—"} bars · train ≤ {signals?.train_cutoff ?? "—"}</span></div>
        <div className="sell-explain">
          <div className="sell-rule">
            <strong>Entry</strong> A direction-agnostic magnitude model flags when NIFTY looks likely to move
            more than {signals?.move_threshold_pct ?? "N"}% over the next {signals?.horizon_bars ?? "N"} five-minute bars.
            Live, this drives a regime switch — a defined-risk credit spread when no big move is expected, a naked
            ATM CE/PE (direction hint, low-confidence) when one is — see the Regime panel above/live poller for the
            actual recommended trade. This replay checks the model + exit logic against history (predicted vs
            realized move %), not a specific option structure's premium P&L.
          </div>
          <div className="sell-rule"><strong>Exit</strong> {signals?.exit_rule ?? "not finalized yet"} — direction-lock once
            price moves away from entry, erosion (giveback) exit once the move stalls, or an S/R zone/session-close/
            max-hold backstop. Same state machine production/nifty_live_poller.py runs live.</div>
          {bt ? (
            <div className="sell-bt">{bt.n} signals ({bt.n_resolved} resolved) · hit rate (|realized| ≥ {signals?.move_threshold_pct}%)
              <b> {(bt.hit_rate * 100).toFixed(0)}%</b> · mean |realized move| <b>{bt.mean_abs_realized_move_pct}%</b>
              <span className="hint"> · out-of-sample from {signals?.train_cutoff} onward — not look-ahead.</span></div>
          ) : <div className="sell-bt"><span className="hint">No backtest results yet — run experiments/nifty_intraday_breakout_v1/gen_indices_signals.py.</span></div>}
        </div>
        <div className="panel-title-controls" style={{ padding: "0 16px 8px" }}>
          <label className="chk"><LineChart size={15} />
            <select value={tf} onChange={(e) => setTf(e.target.value as NiftyTF)} style={{ minWidth: 90 }}>
              {(["5m", "15m", "1D", "1W"] as NiftyTF[]).map((t) => <option key={t} value={t}>{NIFTY_TF_LABEL[t]}</option>)}
            </select>
          </label>
          {isIntraday ? (
            <label className="chk"><CalendarDays size={15} />
              <select value={date} onChange={(e) => setDate(e.target.value)} style={{ minWidth: 140 }}>
                {dateGroups.length ? dateGroups.map(([ym, ds]) => (
                  <optgroup key={ym} label={monthLabel(ym)}>
                    {ds.map((d) => <option key={d} value={d}>{d}{d === new Date().toLocaleDateString("en-CA") ? " (today)" : ""}</option>)}
                  </optgroup>
                )) : <option value="">No trading days yet</option>}
              </select>
            </label>
          ) : <span className="hint">{chartBars.length} {tf === "1D" ? "days" : "weeks"}</span>}
        </div>
        {chartBars.length ? (
          <IntradayCandles series={chartBars} trades={rangeTrades} tf={tf} dailyFull={dailyBars} weeklyFull={weeklyBars}
                           intraZoneSource={bars15mRange} moveThresholdPct={signals?.move_threshold_pct ?? 0.25} />
        ) : <p className="hint" style={{ padding: 16 }}>
          {isIntraday ? (date ? "No 5-min bars ending this date." : "Loading…") : "No daily history yet."}</p>}
      </section>

      <section className="panel cockpit" style={{ marginTop: 14 }}>
        <div className="panel-title"><h2>Signals — trailing 30 days ending {date || "—"}</h2></div>
        <div className="table-wrap">
          <table>
            <thead><tr>
              <th>Entry</th><th>Exit</th><th>Entry NIFTY</th><th>Exit NIFTY</th>
              <th>Predicted move</th><th>Realized move</th><th>Direction</th><th>Exit reason</th>
            </tr></thead>
            <tbody>
              {rangeTrades.map((t, i) => (
                <tr key={i}>
                  <td>{t.entry_ts.slice(5, 16).replace("T", " ")}</td>
                  <td>{t.exit_ts ? t.exit_ts.slice(5, 16).replace("T", " ") : "—"}</td>
                  <td>{num(t.entry_underlying, 1)}</td>
                  <td>{t.exit_underlying != null ? num(t.exit_underlying, 1) : "—"}</td>
                  <td>{t.predicted_move_pct.toFixed(2)}%</td>
                  <td className={t.realized_move_pct == null ? "" : t.realized_move_pct >= 0 ? "move-up" : "move-down"}>
                    {t.realized_move_pct != null ? `${t.realized_move_pct >= 0 ? "+" : ""}${t.realized_move_pct.toFixed(2)}%` : "open"}
                  </td>
                  <td className="hint">{t.direction ?? "—"}</td>
                  <td className="hint">{t.exit_reason ?? "—"}</td>
                </tr>
              ))}
              {!rangeTrades.length && <tr><td colSpan={8} className="empty-cell">No signals in this range</td></tr>}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}

// ----------------------------------------------------------------- Run / Retrain
function fmtTime(t: number | null | undefined): string { return t ? new Date(t * 1000).toLocaleString() : "—"; }
function JobTail({ text, live }: { text: string; live: boolean }) {
  const ref = useRef<HTMLPreElement | null>(null);
  useEffect(() => { if (ref.current) ref.current.scrollTop = ref.current.scrollHeight; }, [text]);
  if (!text) return <span className="hint">{live ? "starting…" : ""}</span>;
  return (
    <pre ref={ref} className={`job-tail${live ? " job-tail-live" : ""}`}>{text}</pre>
  );
}

function RunRetrain() {
  const [st, setSt] = useState<Status | null>(null);
  const [nd, setNd] = useState<{ rows?: number; modified?: number } | null>(null);
  const [msg, setMsg] = useState("");
  const today = new Date().toLocaleDateString("en-CA");
  const [start, setStart] = useState(today);
  const [end, setEnd] = useState(today);
  async function refresh() {
    try {
      const [a, b] = await Promise.all([getJson<Status>("/prod2/status"), getJson<{ rows?: number; modified?: number }>("/prod2/nextday/status")]);
      setSt(a); setNd(b);
    } catch (e) { setMsg(`${(e as Error).message}`); }
  }
  const jobs = st?.jobs ?? {};
  const anyRunning = Object.values(jobs).some((v) => v.status === "running");
  useEffect(() => {
    refresh();
    const t = setInterval(refresh, anyRunning ? 1200 : 4000);
    return () => clearInterval(t);
  }, [anyRunning]);
  async function run(path: string, label: string) {
    setMsg(`starting ${label}…`);
    try { const r = await postJson<{ status: string }>(path); setMsg(`${label}: ${r.status}`); setTimeout(refresh, 500); }
    catch (e) { setMsg(`error: ${(e as Error).message}`); }
  }
  return (
    <>
      <section className="controls-band">
        <label className="hint" htmlFor="refresh-from">From</label>
        <input id="refresh-from" type="date" value={start} max={end} onChange={(e) => setStart(e.target.value)} />
        <label className="hint" htmlFor="refresh-to">To</label>
        <input id="refresh-to" type="date" value={end} min={start} max={today} onChange={(e) => setEnd(e.target.value)} />
        <button type="button" className="btn-wide" onClick={() => run(`/prod2/refresh?start=${start}&end=${end}`, `refresh ${start}→${end}`)}><Play size={16} /> Refresh</button>
        <button type="button" onClick={refresh} title="Refresh status"><RefreshCw size={18} /></button>
        <span className="hint">{msg}</span>
      </section>
      <section className="ops-grid">
        <div className="ops-card"><h4>5-day mover book</h4><strong>{st?.book?.rows ?? "—"}</strong> picks<br /><span className="hint">built {fmtTime(st?.book?.modified)}</span></div>
        <div className="ops-card"><h4>1-day book</h4><strong>{nd?.rows ?? "—"}</strong> rows<br /><span className="hint">built {fmtTime(nd?.modified)}</span></div>
        <div className="ops-card"><h4>Premium OHLC</h4><strong>{st?.premiums?.rows ?? "—"}</strong> picks<br /><span className="hint">built {fmtTime(st?.premiums?.modified)}</span></div>
        <div className="ops-card"><h4>Engine</h4><strong>{st?.version ?? "—"}</strong><br /><span className="hint">rank {String((st?.selector as any)?.ranker ?? "atm_iv")} · top-{String((st?.selector as any)?.picks_per_group_per_day ?? 3)}/grp</span></div>
      </section>
      <section className="panel cockpit" style={{ marginTop: 14 }}>
        <div className="panel-title"><h2>Jobs</h2><span>{anyRunning ? "live · auto-refresh 1.2s" : "auto-refresh 4s"}</span></div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>job</th><th>module</th><th>status</th><th>output (tail)</th></tr></thead>
            <tbody>
              {Object.entries(jobs).map(([k, v]) => (
                <tr key={k}><td>{k}</td><td className="byyear">{v.module ?? "—"}</td>
                  <td><span className={`pass ${v.status === "done" ? "yes" : v.status === "failed" ? "no" : "live"}`}>{v.status}</span></td>
                  <td className="byyear"><JobTail text={v.tail ?? ""} live={v.status === "running"} /></td></tr>
              ))}
              {!Object.keys(jobs).length && <tr><td colSpan={4} className="empty-cell">No runs this session</td></tr>}
            </tbody>
          </table>
        </div>
        <p className="hint" style={{ padding: "8px 16px" }}><strong>Refresh</strong> runs the full pipeline from the selected date through today — fetch bhavcopy + FII → silver → features → both books (minutes). Does not change the locked PROD config.</p>
      </section>
    </>
  );
}

function MoveCell({ p }: { p: Pick }) {
  if (p.live || p.move_mag_pct == null) return <span className="hint">pending</span>;
  const up = Number(p.up_move ?? 0) >= Number(p.down_move ?? 0);
  return <span className={up ? "move-up" : "move-down"}>{up ? "+" : "−"}{num(p.move_mag_pct)}%</span>;
}
function Ohlc({ o, h, l, c, m }: { o?: number | null; h?: number | null; l?: number | null; c?: number | null; m?: number | null }) {
  if (o == null) return <>—</>;
  return <>{num(o)} / {num(h)} / {num(l)} / {num(c)} {m != null ? <span className="thr-badge">{num(m, 2)}x</span> : null}</>;
}

const rootElement = document.getElementById("root");
if (rootElement) createRoot(rootElement).render(<App />);
