import { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { ArrowDown, ArrowUp, ArrowUpDown, CalendarDays, Cog, Filter, Flame, History, LineChart, Lock, Play, RefreshCw } from "lucide-react";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8003";
const GROUP_LABEL: Record<string, string> = { A_mcap30: "A · mega-cap", B_turn35: "B · movers" };
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
type PricePoint = { date: string; open: number; close: number; high: number; low: number; picked: boolean };
type Timeframe = "D" | "W" | "M";
const TF_LABEL: Record<Timeframe, string> = { D: "Daily", W: "Weekly", M: "Monthly" };
type Status = { book?: { modified: number | null; rows: number | null }; premiums?: { modified: number | null; rows: number | null }; version?: string; selector?: Record<string, unknown>; jobs: Record<string, { status: string; module?: string; tail?: string }> };

function pct(v: unknown, d = 1): string { const n = Number(v ?? 0); return Number.isFinite(n) ? `${n.toFixed(d)}%` : "-"; }
function num(v: unknown, d = 1): string { const n = Number(v); return Number.isFinite(n) ? n.toFixed(d) : "—"; }
async function getJson<T>(p: string): Promise<T> { const r = await fetch(`${API_BASE}${p}`); if (!r.ok) throw new Error(`${r.status} ${r.statusText}`); return r.json(); }
async function postJson<T>(p: string): Promise<T> { const r = await fetch(`${API_BASE}${p}`, { method: "POST" }); if (!r.ok) throw new Error(`${r.status}`); return r.json(); }

const TABS = [
  { id: "desk", label: "Signal Desk", icon: <Flame size={16} /> },
  { id: "movers", label: "Universe", icon: <Filter size={16} /> },
  { id: "price", label: "Stock Detail", icon: <LineChart size={16} /> },
  { id: "ops", label: "Refresh / Retrain", icon: <Cog size={16} /> }
];

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
  const [tab, setTab] = useState("desk");
  const [horizon, setHorizon] = useState<Horizon>("5d");
  const [version, setVersion] = useState("prod");
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
      {tab === "desk" && <SignalDesk {...props} />}
      {tab === "movers" && <DailyMovers {...props} />}
      {tab === "price" && <PriceHistory {...props} />}
      {tab === "ops" && <RunRetrain />}
    </main>
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
  useEffect(() => {
    getJson<string[]>(`/prod3/dates?horizon=${horizon}`).then((d) => { setDates(d); setDate((old) => d.includes(old) ? old : d[0] || ""); }).catch((e) => setErr(String(e)));
  }, [horizon]);
  useEffect(() => {
    if (!date) return;
    setData(null);
    getJson<DeskResp>(`/signal-desk?date=${date}&horizon=${horizon}`).then(setData).catch((e) => setErr(String(e)));
  }, [date, horizon]);
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
          <td><span className="group-tag">{GROUP_LABEL[r.group] ?? r.group}</span></td><td>#{r.rank}</td><td className="symbol">{r.symbol}</td><td>{r.conv_pctile != null ? `${(r.conv_pctile * 100).toFixed(0)}th` : "—"}</td><td>{pct(r.atm_iv * 100, 0)}</td><td>{r.atm2_contracts ? `${Math.round(r.atm2_contracts).toLocaleString()} contracts` : "—"}</td>
          <td>{r.v2_pick ? <span className="thr-badge">pick #{r.pick_rank}</span> : <span className="hint">not v2 pick</span>}</td>
          {horizon === "5d" ? <><td><strong>{r.pred_move_pct != null ? pct(r.pred_move_pct, 2) : "—"}</strong></td><td><SignedActual value={r.actual_peak_signed_pct} live={r.live} /></td><td>{r.lean ? <span className="thr-badge">{r.lean} · {pct((r.dir_conf ?? 0) * 100, 0)}</span> : <span className="hint">agnostic</span>}</td></> : <><td><strong>{r.pred_move_pct != null ? pct(r.pred_move_pct, 2) : "—"}</strong></td><td><SignedActual value={r.actual_peak_signed_pct} live={r.live} /></td><td>{r.next_day_peak_expected_pct != null ? <>{pct(r.next_day_peak_expected_pct, 2)} / <SignedActual value={r.next_day_peak_signed_actual_pct} live={r.live} /></> : "—"}</td><td>{r.next_day_close_expected_pct != null ? <>{pct(r.next_day_close_expected_pct, 2)} / <SignedActual value={r.next_day_close_signed_actual_pct} live={r.live} /></> : "—"}</td></>}
        </tr>)}{!rows.length && <tr><td className="empty-cell" colSpan={horizon === "5d" ? 10 : 11}>Loading or no signals</td></tr>}</tbody></table></div>
    </section>
    <section className="panel cockpit" style={{ marginTop: 14 }}><div className="panel-title"><h2>Production decision policy</h2><span>guardrail</span></div><p className="hint" style={{ padding: "4px 16px 16px" }}>{data?.decision?.promotion_gate ?? "Scorecard unavailable — build the production scorecard after refresh."}</p></section>
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
function PriceHistory({ horizon, setHorizon }: PageProps) {
  const [symbols, setSymbols] = useState<{ symbol: string; group: string }[]>([]);
  const [symbol, setSymbol] = useState("ADANIENT");
  const [data, setData] = useState<{ series: PricePoint[]; premiums: Pick[] }>({ series: [], premiums: [] });
  const [nd, setNd] = useState<NDRow[]>([]);
  const [tf, setTf] = useState<Timeframe>("D");
  const [showLevels, setShowLevels] = useState(true);
  const [minDD, setMinDD] = useState(10);
  const one = horizon === "1d";
  const candles = useMemo(() => resample(data.series, tf), [data.series, tf]);
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
        <label><LineChart size={16} />
          <select value={symbol} onChange={(e) => setSymbol(e.target.value)}>
            {symbols.map((s) => <option key={s.symbol} value={s.symbol}>{s.symbol} ({GROUP_LABEL[s.group] ?? s.group})</option>)}
          </select>
        </label>
        <label><CalendarDays size={16} />
          <select value={tf} onChange={(e) => setTf(e.target.value as Timeframe)}>
            {(["D", "W", "M"] as Timeframe[]).map((t) => <option key={t} value={t}>{TF_LABEL[t]}</option>)}
          </select>
        </label>
        <label className="chk">
          <input type="checkbox" checked={showLevels} onChange={(e) => setShowLevels(e.target.checked)} /> S/R levels
        </label>
        <label>DD %
          <input type="number" min={1} max={40} step={1} value={minDD} style={{ width: 54 }}
                 onChange={(e) => setMinDD(Math.max(1, Math.min(40, Number(e.target.value) || 5)))} />
        </label>
        <span className="hint">latest {DEFAULT_CANDLES} · scroll to zoom · ← → to pan · hover for OHLC · double-click to reset</span>
      </section>
      <section className="panel cockpit">
        <div className="panel-title"><h2>{symbol} — {TF_LABEL[tf].toLowerCase()} candles</h2><span>{candles.length} {tf === "D" ? "days" : tf === "W" ? "weeks" : "months"}</span></div>
        <div style={{ padding: 14 }}><Candles series={candles} showLevels={showLevels} minDDpct={minDD} /></div>
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
      picked: arr.some((p) => p.picked),
    });
  }
  return out;
}

type Level = { price: number; kind: "R" | "S" | "ATH"; touches: number; firstIdx: number };

// Wick-based support/resistance, evaluated as of the LAST bar in `series` (so panning
// re-derives them from the data visible up to that day). N-bar pivot highs/lows are
// clustered by price; a cluster is a zone only if the level was revisited >= 2 times
// with a real pullback (>= minDD) BETWEEN visits (a flat stall = one touch). Returns at
// most 3: nearest support below the last close, nearest resistance above it, and — if a
// zone sits at the all-time high — that ATH zone (kind "ATH"). firstIdx = first touch.
function srLevels(series: PricePoint[], minDD: number): Level[] {
  const N = 3, tol = 0.008, minTouches = 2, athTol = 0.02;
  const n = series.length;
  if (n < 2 * N + 1) return [];
  const hiP: number[] = [], loP: number[] = [];
  for (let i = N; i < n - N; i++) {
    let isHi = true, isLo = true;
    for (let j = i - N; j <= i + N; j++) {
      if (j === i) continue;
      if (series[j].high >= series[i].high) isHi = false;
      if (series[j].low <= series[i].low) isLo = false;
    }
    if (isHi) hiP.push(i);
    if (isLo) loP.push(i);
  }
  const build = (idx: number[], kind: "R" | "S"): Level[] => {
    const pts = idx.map((i) => ({ i, p: kind === "R" ? series[i].high : series[i].low }))
                   .sort((a, b) => a.p - b.p);
    const out: Level[] = [];
    let cl: { i: number; p: number }[] = [];
    const flush = () => {
      if (cl.length) {
        const level = cl.reduce((s, x) => s + x.p, 0) / cl.length;
        const byTime = cl.slice().sort((a, b) => a.i - b.i);
        let touches = 0, ref = -1;
        for (const pt of byTime) {
          if (ref < 0) { touches = 1; ref = pt.i; continue; }
          let pulled: boolean;
          if (kind === "R") {
            let mn = Infinity;
            for (let k = ref; k <= pt.i; k++) mn = Math.min(mn, series[k].low);
            pulled = mn <= level * (1 - minDD);
          } else {
            let mx = -Infinity;
            for (let k = ref; k <= pt.i; k++) mx = Math.max(mx, series[k].high);
            pulled = mx >= level * (1 + minDD);
          }
          if (pulled) touches++;
          ref = pt.i;
        }
        if (touches >= minTouches) out.push({ price: level, kind, touches, firstIdx: byTime[0].i });
      }
      cl = [];
    };
    for (const pt of pts) {
      if (cl.length && Math.abs(pt.p - cl[cl.length - 1].p) / pt.p > tol) flush();
      cl.push(pt);
    }
    flush();
    return out;
  };
  // all zones, merge overlapping highs/lows
  const all = [...build(hiP, "R"), ...build(loP, "S")].sort((a, b) => a.price - b.price);
  const merged: Level[] = [];
  for (const L of all) {
    const last = merged[merged.length - 1];
    if (last && Math.abs(L.price - last.price) / L.price <= tol) {
      const tot = last.touches + L.touches;
      last.price = (last.price * last.touches + L.price * L.touches) / tot;
      last.touches = Math.max(last.touches, L.touches);
      last.firstIdx = Math.min(last.firstIdx, L.firstIdx);
    } else merged.push({ ...L });
  }
  // select: nearest support below + nearest resistance above the last visible close,
  // plus the all-time-high zone (blue) if one exists — max 3 lines
  const refClose = series[n - 1].close;
  let ath = -Infinity;
  for (const p of series) ath = Math.max(ath, p.high);
  const support = merged.filter((L) => L.price < refClose).sort((a, b) => b.price - a.price)[0];
  const resistance = merged.filter((L) => L.price > refClose).sort((a, b) => a.price - b.price)[0];
  const athZone = merged.filter((L) => Math.abs(L.price - ath) / ath <= athTol).sort((a, b) => b.price - a.price)[0];
  const sel: Level[] = [];
  const add = (L: Level | undefined, isAth = false) => {
    if (!L) return;
    const dup = sel.find((x) => Math.abs(x.price - L.price) / L.price < 1e-9);
    if (dup) { if (isAth) dup.kind = "ATH"; return; }
    sel.push(isAth ? { ...L, kind: "ATH" } : L);
  };
  add(support);
  add(resistance);
  if (athZone) add(athZone, true);
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

const DEFAULT_CANDLES = 100;

function Candles({ series, showLevels, minDDpct }: { series: PricePoint[]; showLevels: boolean; minDDpct: number }) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const defWin = (len: number) => ({ s: Math.max(0, len - DEFAULT_CANDLES), e: len });
  const [win, setWin] = useState<{ s: number; e: number }>(defWin(series.length));
  const [hover, setHover] = useState<number | null>(null);
  // default to the most recent ~100 candles; reset when the series changes (symbol / timeframe)
  useEffect(() => { setWin(defWin(series.length)); setHover(null); }, [series]);

  const W = 900, H = 300, padX = 40, padTop = 14, padBot = 34;
  const total = series.length;
  const s = Math.max(0, Math.min(win.s, Math.max(0, total - 2)));
  const e = Math.max(s + 2, Math.min(win.e, total));
  const vis = series.slice(s, e);
  const n = vis.length;
  const slot = n > 0 ? (W - 2 * padX) / n : 0;
  const cw = Math.max(1, Math.min(14, slot * 0.7));
  const x = (i: number) => padX + slot * (i + 0.5);

  // S/R levels re-derived as of the last VISIBLE bar (series[0..e)); recompute on pan/zoom
  const levels = useMemo(
    () => (showLevels ? srLevels(series.slice(0, e), minDDpct / 100) : []),
    [series, e, showLevels, minDDpct],
  );

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

  // keyboard: left / right arrows pan the window (ignored while typing in a field)
  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      const t = ev.target as HTMLElement | null;
      if (t && /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName)) return;
      if (ev.key !== "ArrowLeft" && ev.key !== "ArrowRight") return;
      ev.preventDefault();
      const width = e - s;
      const step = Math.max(1, Math.round(width * 0.1));
      if (ev.key === "ArrowLeft") { const ns = Math.max(0, s - step); setWin({ s: ns, e: ns + width }); }
      else { const ne = Math.min(total, e + step); setWin({ s: ne - width, e: ne }); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [s, e, total]);

  if (total < 2) return <span className="hint">No data</span>;

  const lvlPrices = levels.map((L) => L.price);
  const hi = Math.max(...vis.map((p) => p.high), ...lvlPrices);
  const lo = Math.min(...vis.map((p) => p.low), ...lvlPrices);
  const y = (v: number) => padTop + (1 - (v - lo) / (hi - lo || 1)) * (H - padTop - padBot);
  const up = "#167c80", down = "#9a2431";
  const atDefault = s === Math.max(0, total - DEFAULT_CANDLES) && e === total;
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

  const hp = hover != null && hover < n ? vis[hover] : null;
  const tipLeftPct = hover != null ? (x(hover) / W) * 100 : 0;
  const tipRight = tipLeftPct > 62;

  return (
    <div ref={wrapRef} className="candles-wrap" style={{ position: "relative" }}
         onMouseMove={(ev) => setHover(idxFromClientX(ev.clientX))}
         onMouseLeave={() => setHover(null)}
         onDoubleClick={() => setWin(defWin(total))}>
      <div className="candles-toolbar">
        <button type="button" title="Zoom in" onClick={() => { const w = Math.max(6, Math.round(n * 0.7)); const mid = s + n / 2; const ns = Math.max(0, Math.min(total - w, Math.round(mid - w / 2))); setWin({ s: ns, e: ns + w }); }}>+</button>
        <button type="button" title="Zoom out" onClick={() => { const w = Math.min(total, Math.round(n * 1.4)); const mid = s + n / 2; const ns = Math.max(0, Math.min(total - w, Math.round(mid - w / 2))); setWin({ s: ns, e: ns + w }); }}>−</button>
        <button type="button" title="Reset to latest 100" disabled={atDefault} onClick={() => setWin(defWin(total))}>Reset</button>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="spark">
        {/* y-axis grid: light-grey lines at rounded price levels */}
        {grid.map((g, gi) => (
          <g key={`grid${gi}`}>
            <line x1={padX} x2={W - padX} y1={y(g)} y2={y(g)} stroke="#e7ecea" strokeWidth={1} />
            <text x={padX - 6} y={y(g) + 3} fontSize={10} fill="#8a978f" textAnchor="end">{num(g, 0)}</text>
          </g>
        ))}
        {hp != null ? <line x1={x(hover!)} x2={x(hover!)} y1={padTop} y2={H - padBot} stroke="#b9c6c0" strokeWidth={1} strokeDasharray="3 3" /> : null}
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
        {/* S/R zones: line runs from its first touch to the right edge; label = price + touches */}
        {levels.map((L, li) => {
          const color = L.kind === "ATH" ? "#2563b0" : "#33443d";
          const xStart = L.firstIdx <= s ? padX : Math.min(W - padX, x(L.firstIdx - s));
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
      {hp ? (
        <div className="candle-tip" style={{ [tipRight ? "right" : "left"]: `calc(${tipRight ? 100 - tipLeftPct : tipLeftPct}% + 10px)`, top: 8 }}>
          <strong>{hp.date}</strong>
          <span>O <b>{num(hp.open, 1)}</b></span>
          <span>H <b>{num(hp.high, 1)}</b></span>
          <span>L <b>{num(hp.low, 1)}</b></span>
          <span>C <b style={{ color: hp.close >= hp.open ? up : down }}>{num(hp.close, 1)}</b></span>
        </div>
      ) : null}
    </div>
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
