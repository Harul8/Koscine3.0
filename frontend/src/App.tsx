import { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { ArrowDown, ArrowUp, ArrowUpDown, CalendarDays, Coins, Cog, ExternalLink, Filter, Flame, History, LineChart, Lock, Play, RefreshCw, X } from "lucide-react";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8003";
const GROUP_LABEL: Record<string, string> = { A_mcap30: "A · mega-cap", B_turn35: "B · movers" };
const MIN_PROPOSAL_ROR_PCT = 150; // top-3 daily proposals never include a return-on-risk below this
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
      {tab === "sell" && <><SellStrategies /><SkewStrategy /></>}
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
function srLevels(vis: PricePoint[], dd: number, ath: number): Level[] {
  // N=2: needs to beat only its 2 nearest neighbors each side. N=3 was too wide — two genuinely
  // separate nearby peaks (e.g. 3 weeks apart on a weekly chart) could sit within each other's
  // pivot window and invalidate one another even though a real pullback separated them.
  const N = 2, tol = 0.02, athTol = 0.02, G = 8;
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
  // Collapse same-side zones within 5% of each other into one, keeping whichever is nearer to
  // the current price: the LOWER price for resistance, the UPPER price for support — so the
  // final displayed lines are never two near-duplicates <5% apart.
  const dedupBySide = (zones: Zone[], keepHighest: boolean): Zone[] => {
    const sorted = zones.slice().sort((a, b) => a.price - b.price);
    const out: Zone[] = [];
    let cl: Zone[] = [];
    const flush = () => {
      if (cl.length) out.push(keepHighest ? cl.reduce((m, z) => (z.price > m.price ? z : m)) : cl.reduce((m, z) => (z.price < m.price ? z : m)));
      cl = [];
    };
    for (const z of sorted) { if (cl.length && Math.abs(z.price - cl[cl.length - 1].price) / z.price > 0.05) flush(); cl.push(z); }
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

  // keyboard: left/right arrows pan, up/down arrows zoom (ignored while typing in a field)
  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      const t = ev.target as HTMLElement | null;
      if (t && /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName)) return;
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
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
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
    <div ref={wrapRef} className="candles-wrap" style={{ position: "relative" }}
         onClick={(ev) => { const idx = idxFromClientX(ev.clientX); setSel((prev) => (prev === idx ? null : idx)); }}
         onDoubleClick={() => { setWin(defWin(total)); setSel(null); }}>
      <div className="candles-toolbar">
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
  sell_premium: number; buy_premium: number;
  credit: number; max_risk: number; max_profit: number; ror_pct: number; be_low: number; be_high: number; in_window: boolean;
};
type SellResp = {
  as_of: string | null; params: Record<string, number>;
  backtest: { window: string; ev_on_risk_all: number; ev_on_risk_iv_rich: number; win_rate: number; worst: string };
  candidates: CondorRow[];
};
type SellHistRow = {
  symbol: string; group: string; signal_date: string; expiry: string; dte: number; iv_ratio: number;
  short_ce: number; long_ce: number; short_pe: number; long_pe: number;
  sell_premium: number; buy_premium: number;
  credit: number; max_risk: number; exit_value: number;
  pnl: number; ror_pct: number; max_dd_pct: number; outcome: string;
};
type SellHist = {
  rows: SellHistRow[];
  summary: { n: number; win_rate: number; ev_ror_pct: number; median_ror_pct: number; worst_ror_pct: number; worst_dd_pct: number; total_pnl: number } | null;
};

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
  const cands = data?.candidates ?? [];
  // top 3 per group for the daily signals
  const daily = ["A_mcap30", "B_turn35"].flatMap((g) => cands.filter((c) => c.group === g && c.ror_pct >= MIN_PROPOSAL_ROR_PCT).slice(0, 3));
  const allSyms = filterSym === "";
  const months = useMemo(() => monthsOf(hist?.rows ?? [], (r) => r.signal_date), [hist]);
  const histRows = useMemo(() => {
    const all = hist?.rows ?? [];
    return filterMonth ? all.filter((r) => r.signal_date.startsWith(filterMonth)) : all;
  }, [hist, filterMonth]);
  const hs = useMemo(() => {
    if (!histRows.length) return null;
    return {
      n: histRows.length, win_rate: histRows.filter((r) => r.outcome === "win").length / histRows.length,
      ev_ror_pct: Math.round((histRows.reduce((s, r) => s + r.ror_pct, 0) / histRows.length) * 10) / 10,
      median_ror_pct: Math.round(median(histRows.map((r) => r.ror_pct)) * 10) / 10,
      worst_ror_pct: Math.round(Math.min(...histRows.map((r) => r.ror_pct)) * 10) / 10,
      worst_dd_pct: Math.round(Math.min(...histRows.map((r) => r.max_dd_pct)) * 10) / 10,
      total_pnl: Math.round(histRows.reduce((s, r) => s + r.pnl, 0) * 10) / 10,
    };
  }, [histRows]);

  return (
    <>
      <section className="panel cockpit">
        <div className="panel-title"><h2>Sell Strategies — defined-risk iron condor</h2>
          <span>as of {data?.as_of ?? "—"} · top 3 per group · ret/risk ≥{MIN_PROPOSAL_ROR_PCT}%</span></div>
        <div className="sell-explain">
          <div className="sell-rule">
            <strong>Structure</strong> Sell the ~2% OTM call & put, buy the ±5% wings — a delta-neutral iron condor.
            <b> Max loss is always capped</b> at (wing width − credit); you can never lose more than the max-risk shown.
          </div>
          <div className="sell-rule"><strong>Signal / entry</strong> Take it when IV is <b>rich</b> (atm-IV above its 1-yr median, iv_ratio ≥ 1.1)
            and the near expiry is <b>≤ ~2 weeks out</b> (DTE ≤ 14). Rows meeting both are the <b>entry-window</b> picks (highlighted).</div>
          <div className="sell-rule"><strong>Exit</strong> Close at <b>~50% of max profit</b> or by expiry (whichever first); it decays in your favour while price stays between the breakevens.</div>
          {bt ? (
            <div className="sell-bt">Backtest ({bt.window}): <b>+{(bt.ev_on_risk_all * 100).toFixed(0)}%</b> mean return-on-risk (all),
              <b> +{(bt.ev_on_risk_iv_rich * 100).toFixed(0)}%</b> when IV-rich, win rate <b>{(bt.win_rate * 100).toFixed(0)}%</b>, worst <b>{bt.worst}</b>.
              <span className="hint"> Gross of costs/STT — model these before sizing up.</span></div>
          ) : null}
        </div>
        <div className="table-wrap">
          <table>
            <thead><tr>
              <th>Symbol</th><th>Grp</th><th>Exp / DTE</th><th>Spot</th><th>IV rich</th>
              <th>Short C / Long / BE</th><th>Short P / Long / BE</th><th>Credit</th><th>Max risk</th><th>Ret/risk</th>
            </tr></thead>
            <tbody>
              {daily.map((r) => (
                <tr key={r.symbol} className={r.in_window ? "sell-live" : ""}>
                  <td><SymbolLink symbol={r.symbol} onOpen={setOpenSymbol} /></td>
                  <td>{r.group.slice(0, 1)}</td>
                  <td>{r.expiry.slice(5)} · {r.dte}d</td>
                  <td>{num(r.underlying, 0)}</td>
                  <td>{r.iv_ratio != null ? <span className={r.iv_ratio >= 1.1 ? "move-up" : "hint"}>{r.iv_ratio.toFixed(2)}×</span> : "—"}</td>
                  <td>{num(r.short_ce, 0)} / {num(r.long_ce, 0)} / <span className="hint">{num(r.be_high, 0)}</span></td>
                  <td>{num(r.short_pe, 0)} / {num(r.long_pe, 0)} / <span className="hint">{num(r.be_low, 0)}</span></td>
                  <td>{num(r.credit, 1)} <span className="hint">(sell {num(r.sell_premium, 1)} / buy {num(r.buy_premium, 1)})</span></td>
                  <td>{num(r.max_risk, 1)} <span className="hint">(width {num(Math.max(r.long_ce - r.short_ce, r.short_pe - r.long_pe), 1)} / credit {num(r.credit, 1)})</span></td>
                  <td><strong>{r.ror_pct.toFixed(0)}%</strong></td>
                </tr>
              ))}
              {!daily.length && <tr><td colSpan={10} className="empty-cell">No candidate clears the {MIN_PROPOSAL_ROR_PCT}% ret/risk bar today</td></tr>}
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
            worst trade <b>{hs.worst_ror_pct}%</b> · worst drawdown <b>{hs.worst_dd_pct}%</b> of risk</div>
        ) : null}
        <div className="table-wrap">
          <table>
            <thead><tr>
              <th>Signal date</th>{allSyms ? <th>Symbol</th> : null}<th>Exp / DTE</th><th>IV</th>
              <th>Short C / P</th><th>Credit</th><th>Max risk</th><th>Exit value</th><th>PnL</th><th>Ret/risk</th><th>Max DD</th><th></th>
            </tr></thead>
            <tbody>
              {histRows.map((r, i) => (
                <tr key={`${r.symbol}-${r.signal_date}-${i}`}>
                  <td>{r.signal_date}</td>
                  {allSyms ? <td><SymbolLink symbol={r.symbol} onOpen={setOpenSymbol} /> <span className="hint">{r.group.slice(0, 1)}</span></td> : null}
                  <td>{r.expiry.slice(5)} · {r.dte}d</td>
                  <td>{r.iv_ratio.toFixed(2)}×</td>
                  <td>{num(r.short_ce, 0)} / {num(r.short_pe, 0)}</td>
                  <td>{num(r.credit, 1)} <span className="hint">(sell {num(r.sell_premium, 1)} / buy {num(r.buy_premium, 1)})</span></td>
                  <td>{num(r.max_risk, 1)} <span className="hint">(width {num(Math.max(r.long_ce - r.short_ce, r.short_pe - r.long_pe), 1)} / credit {num(r.credit, 1)})</span></td>
                  <td>{num(r.exit_value, 1)}</td>
                  <td className={r.pnl >= 0 ? "move-up" : "move-down"}>{r.pnl >= 0 ? "+" : ""}{num(r.pnl, 1)}</td>
                  <td className={r.ror_pct >= 0 ? "move-up" : "move-down"}>{r.ror_pct >= 0 ? "+" : ""}{r.ror_pct.toFixed(0)}%</td>
                  <td className="move-down">{r.max_dd_pct.toFixed(0)}%</td>
                  <td>{r.outcome === "win" ? "✓" : "✕"}</td>
                </tr>
              ))}
              {!histRows.length && <tr><td colSpan={allSyms ? 12 : 11} className="empty-cell">No signals</td></tr>}
            </tbody>
          </table>
        </div>
      </section>
      {openSymbol && <ChartModal symbol={openSymbol} onClose={() => setOpenSymbol(null)} />}
    </>
  );
}

// ----------------------------------------------------------------- Skew Strategy (directional leg)
type SkewRow = {
  symbol: string; group: string; expiry: string; dte: number; underlying: number; iv_ratio: number | null;
  side: "CE" | "PE"; ce_iv: number; pe_iv: number; skew: number;
  short_strike: number; long_strike: number; sell_premium: number; buy_premium: number;
  credit: number; max_risk: number; max_profit: number;
  ror_pct: number; breakeven: number; in_window: boolean;
};
type SkewResp = {
  as_of: string | null; params: Record<string, number>;
  backtest: { window: string; ev_on_risk: number; win_rate: number; worst: string; note: string };
  candidates: SkewRow[];
};
type SkewHistRow = {
  symbol: string; group: string; signal_date: string; expiry: string; dte: number; iv_ratio: number;
  side: "CE" | "PE"; ce_iv: number; pe_iv: number; skew: number; short_strike: number; long_strike: number;
  sell_premium: number; buy_premium: number;
  credit: number; max_risk: number; exit_value: number; pnl: number; ror_pct: number; max_dd_pct: number; outcome: string;
};
type SkewHist = {
  rows: SkewHistRow[];
  summary: { n: number; win_rate: number; ev_ror_pct: number; median_ror_pct: number; worst_ror_pct: number; worst_dd_pct: number; total_pnl: number } | null;
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
  const cands = data?.candidates ?? [];
  const daily = ["A_mcap30", "B_turn35"].flatMap((g) => cands.filter((c) => c.group === g && c.ror_pct >= MIN_PROPOSAL_ROR_PCT).slice(0, 3));
  const allSyms = filterSym === "";
  const months = useMemo(() => monthsOf(hist?.rows ?? [], (r) => r.signal_date), [hist]);
  const histRows = useMemo(() => {
    const all = hist?.rows ?? [];
    return filterMonth ? all.filter((r) => r.signal_date.startsWith(filterMonth)) : all;
  }, [hist, filterMonth]);
  const hs = useMemo(() => {
    if (!histRows.length) return null;
    return {
      n: histRows.length, win_rate: histRows.filter((r) => r.outcome === "win").length / histRows.length,
      ev_ror_pct: Math.round((histRows.reduce((s, r) => s + r.ror_pct, 0) / histRows.length) * 10) / 10,
      median_ror_pct: Math.round(median(histRows.map((r) => r.ror_pct)) * 10) / 10,
      worst_ror_pct: Math.round(Math.min(...histRows.map((r) => r.ror_pct)) * 10) / 10,
      worst_dd_pct: Math.round(Math.min(...histRows.map((r) => r.max_dd_pct)) * 10) / 10,
      total_pnl: Math.round(histRows.reduce((s, r) => s + r.pnl, 0) * 10) / 10,
    };
  }, [histRows]);

  return (
    <>
      <section className="panel cockpit" style={{ marginTop: 14 }}>
        <div className="panel-title"><h2>Directional credit spread — IV skew</h2>
          <span>as of {data?.as_of ?? "—"} · top 3 per group · ret/risk ≥{MIN_PROPOSAL_ROR_PCT}%</span></div>
        <div className="sell-explain">
          <div className="sell-rule">
            <strong>Structure</strong> Sell only the side (call or put) whose ~2% OTM strike is priced with the <b>richer</b> Black-Scholes implied vol
            (skew = CE-IV − PE-IV), buy the +5% wing on that same side — a single-leg defined-risk credit spread.
            <b> Max loss is always capped</b> at (wing width − credit).
            <span className="hint"> This is a relative-value read on option pricing, not a forecast of which way the stock moves — direction alone is ≈ coin-flip on this universe.</span>
          </div>
          <div className="sell-rule"><strong>Signal / entry</strong> Take it when IV is <b>rich</b> (atm-IV above its 1-yr median, iv_ratio ≥ 1.1)
            and the near expiry is <b>≤ ~2 weeks out</b> (DTE ≤ 14). Rows meeting both are the <b>entry-window</b> picks (highlighted).</div>
          <div className="sell-rule"><strong>Exit</strong> Close at <b>~50% of max profit</b> or by expiry (whichever first) — <b>no interim stop-loss</b>.
            Every EOD stop level tested (15–50% of max risk) reduced win rate and mean return: at 5–12 DTE, day-close dips mostly mean-revert by expiry, so a stop just locks in a temporary drawdown.</div>
          {bt ? (
            <div className="sell-bt">Backtest ({bt.window}): <b>+{(bt.ev_on_risk * 100).toFixed(0)}%</b> mean return-on-risk,
              win rate <b>{(bt.win_rate * 100).toFixed(0)}%</b>, worst <b>{bt.worst}</b>.
              <span className="hint"> Gross of costs/STT — model these before sizing up.</span></div>
          ) : null}
        </div>
        <div className="table-wrap">
          <table>
            <thead><tr>
              <th>Symbol</th><th>Grp</th><th>Exp / DTE</th><th>Spot</th><th>IV rich</th><th>Sell</th>
              <th>CE-IV / PE-IV</th><th>Short / Long</th><th>Credit</th><th>Max risk</th><th>Ret/risk</th><th>Breakeven</th>
            </tr></thead>
            <tbody>
              {daily.map((r) => (
                <tr key={r.symbol} className={r.in_window ? "sell-live" : ""}>
                  <td><SymbolLink symbol={r.symbol} onOpen={setOpenSymbol} /></td>
                  <td>{r.group.slice(0, 1)}</td>
                  <td>{r.expiry.slice(5)} · {r.dte}d</td>
                  <td>{num(r.underlying, 0)}</td>
                  <td>{r.iv_ratio != null ? <span className={r.iv_ratio >= 1.1 ? "move-up" : "hint"}>{r.iv_ratio.toFixed(2)}×</span> : "—"}</td>
                  <td><span className={r.side === "CE" ? "side long" : "side short"}>{r.side === "CE" ? "Call" : "Put"}</span></td>
                  <td className="hint">{r.ce_iv.toFixed(2)} / {r.pe_iv.toFixed(2)}</td>
                  <td>{num(r.short_strike, 0)} / {num(r.long_strike, 0)}</td>
                  <td>{num(r.credit, 1)} <span className="hint">(sell {num(r.sell_premium, 1)} / buy {num(r.buy_premium, 1)})</span></td>
                  <td>{num(r.max_risk, 1)} <span className="hint">(width {num(Math.abs(r.long_strike - r.short_strike), 1)} / credit {num(r.credit, 1)})</span></td>
                  <td><strong>{r.ror_pct.toFixed(0)}%</strong></td>
                  <td className="hint">{num(r.breakeven, 0)}</td>
                </tr>
              ))}
              {!daily.length && <tr><td colSpan={12} className="empty-cell">No candidate clears the {MIN_PROPOSAL_ROR_PCT}% ret/risk bar today</td></tr>}
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
            worst trade <b>{hs.worst_ror_pct}%</b> · worst drawdown <b>{hs.worst_dd_pct}%</b> of risk</div>
        ) : null}
        <div className="table-wrap">
          <table>
            <thead><tr>
              <th>Signal date</th>{allSyms ? <th>Symbol</th> : null}<th>Exp / DTE</th><th>IV</th><th>Sell</th>
              <th>Short / Long</th><th>Credit</th><th>Max risk</th><th>Exit value</th><th>PnL</th><th>Ret/risk</th><th>Max DD</th><th></th>
            </tr></thead>
            <tbody>
              {histRows.map((r, i) => (
                <tr key={`${r.symbol}-${r.signal_date}-${i}`}>
                  <td>{r.signal_date}</td>
                  {allSyms ? <td><SymbolLink symbol={r.symbol} onOpen={setOpenSymbol} /> <span className="hint">{r.group.slice(0, 1)}</span></td> : null}
                  <td>{r.expiry.slice(5)} · {r.dte}d</td>
                  <td>{r.iv_ratio.toFixed(2)}×</td>
                  <td><span className={r.side === "CE" ? "side long" : "side short"}>{r.side === "CE" ? "Call" : "Put"}</span></td>
                  <td>{num(r.short_strike, 0)} / {num(r.long_strike, 0)}</td>
                  <td>{num(r.credit, 1)} <span className="hint">(sell {num(r.sell_premium, 1)} / buy {num(r.buy_premium, 1)})</span></td>
                  <td>{num(r.max_risk, 1)} <span className="hint">(width {num(Math.abs(r.long_strike - r.short_strike), 1)} / credit {num(r.credit, 1)})</span></td>
                  <td>{num(r.exit_value, 1)}</td>
                  <td className={r.pnl >= 0 ? "move-up" : "move-down"}>{r.pnl >= 0 ? "+" : ""}{num(r.pnl, 1)}</td>
                  <td className={r.ror_pct >= 0 ? "move-up" : "move-down"}>{r.ror_pct >= 0 ? "+" : ""}{r.ror_pct.toFixed(0)}%</td>
                  <td className="move-down">{r.max_dd_pct.toFixed(0)}%</td>
                  <td>{r.outcome === "win" ? "✓" : "✕"}</td>
                </tr>
              ))}
              {!histRows.length && <tr><td colSpan={allSyms ? 13 : 12} className="empty-cell">No signals</td></tr>}
            </tbody>
          </table>
        </div>
      </section>
      {openSymbol && <ChartModal symbol={openSymbol} onClose={() => setOpenSymbol(null)} />}
    </>
  );
}

// ----------------------------------------------------------------- Run / Retrain
function fmtTime(t: number | null | undefined): string { return t ? new Date(t * 1000).toLocaleString() : "—"; }
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
  useEffect(() => { refresh(); const t = setInterval(refresh, 4000); return () => clearInterval(t); }, []);
  async function run(path: string, label: string) {
    setMsg(`starting ${label}…`);
    try { const r = await postJson<{ status: string }>(path); setMsg(`${label}: ${r.status}`); setTimeout(refresh, 800); }
    catch (e) { setMsg(`error: ${(e as Error).message}`); }
  }
  const jobs = st?.jobs ?? {};
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
        <div className="panel-title"><h2>Jobs</h2><span>auto-refresh 4s</span></div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>job</th><th>module</th><th>status</th><th>output (tail)</th></tr></thead>
            <tbody>
              {Object.entries(jobs).map(([k, v]) => (
                <tr key={k}><td>{k}</td><td className="byyear">{v.module ?? "—"}</td>
                  <td><span className={`pass ${v.status === "done" ? "yes" : v.status === "failed" ? "no" : "live"}`}>{v.status}</span></td>
                  <td className="byyear" style={{ whiteSpace: "pre-wrap", maxWidth: 520 }}>{v.tail ?? ""}</td></tr>
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
