// Koscine 4 trade list and ledger page (requirements O-6, O-7).
// Reads the daily lists the daily run wrote and the ledger; the only writes are fills typed in by
// the owner, "not taken", and closing a trade from its fills. Nothing here trains or backtests.
import { useCallback, useEffect, useMemo, useState } from "react";
import { RefreshCw } from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8003";

type ListIndex = { date: string; decision_time: string; mode: string; new_trades: number; open_trades: number };
type TradeLeg = { action: string; expiry: string; strike: number; type: string; reference_price: number };
type NewTrade = {
  ledger_id?: number; instrument: string; strategy: string; structure: string; legs: TradeLeg[]; lots: number; lot_size: number;
  entry: { date: string; window: string; net_price_limit_per_share: number };
  max_loss_rupees: number; max_loss_per_lot: number; margin_estimate_rupees: number;
  margin_inputs?: { method?: string; price_scan_pct?: number; worst_scenario?: string };
  expected_result_pct_of_max_loss: number | null; expected_result_inputs?: Record<string, unknown>;
  exit_rules: Record<string, unknown>; score: number;
};
type OpenInstruction = { id: number; instrument: string; strategy: string; instruction: string; sessions_held?: number; result_pct_so_far?: number };
type Refused = { instrument: string; strategy: string; score: number; expected_result_pct?: number | null; reason: string };
type Warning = { check: string; instrument: string; file: string; detail: string };
type TradeListDoc = {
  date: string; decision_time: string; mode: string; data_up_to: string; message: string; no_trade_today: boolean;
  new_trades: NewTrade[]; open_trades: OpenInstruction[]; best_refused: Refused[]; data_warnings?: Warning[]; feature_warnings?: unknown[];
};
type LedgerLeg = [string, number, string, number];
type Fill = { id: number; kind: string; leg: number; price: number; quantity: number; filled_at: string; note: string };
type Rec = {
  id: number; decision_date: string; decision_time: string; entry_date: string; strategy: string; book: string; instrument: string;
  legs: LedgerLeg[]; lots: number; lot_size: number; max_loss: number; status: string; mode: string;
  reference_prices: number[]; exit_date?: string | null; exit_reason?: string | null; pnl?: number | null; result_pct?: number | null;
  fills?: Fill[];
};

async function call<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, body === undefined ? undefined : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); if (j?.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch { /* not JSON */ }
    throw new Error(detail);
  }
  return r.json();
}

const rupees = (v: number | null | undefined) => (v == null || !Number.isFinite(v) ? "—" : `₹${Math.round(v).toLocaleString("en-IN")}`);
const fixed = (v: number | null | undefined, d = 2) => (v == null || !Number.isFinite(v) ? "—" : v.toFixed(d));
const legLabel = (l: LedgerLeg) => `${l[3] === 1 ? "BUY" : "SELL"} ${l[0]} ${l[1]} ${l[2]}`;

export function TradeListTab() {
  const [lists, setLists] = useState<ListIndex[]>([]);
  const [chosen, setChosen] = useState("");
  const [doc, setDoc] = useState<TradeListDoc | null>(null);
  const [open, setOpen] = useState<Rec[]>([]);
  const [closed, setClosed] = useState<Rec[]>([]);
  const [selected, setSelected] = useState<Rec | null>(null);
  const [msg, setMsg] = useState("");

  const loadLedger = useCallback(async () => {
    try {
      const [o, c] = await Promise.all([call<Rec[]>("/v1/ledger/open"), call<Rec[]>("/v1/ledger/closed")]);
      setOpen(o); setClosed(c);
    } catch (e) { setMsg(`Ledger: ${(e as Error).message}`); }
  }, []);

  const loadLists = useCallback(async () => {
    try {
      const l = await call<ListIndex[]>("/v1/trade-lists");
      setLists(l);
      setChosen((c) => c || (l.length ? `${l[0].date}|${l[0].decision_time}` : ""));
    } catch (e) { setMsg(`Trade lists: ${(e as Error).message}`); }
  }, []);

  const refresh = useCallback(() => { setMsg(""); loadLists(); loadLedger(); }, [loadLists, loadLedger]);
  useEffect(() => { refresh(); }, [refresh]);

  useEffect(() => {
    if (!chosen) { setDoc(null); return; }
    const [date, time] = chosen.split("|");
    call<TradeListDoc>(`/v1/trade-list/${date}?time=${time}`).then(setDoc).catch((e) => setMsg(`List: ${(e as Error).message}`));
  }, [chosen]);

  const openRec = useCallback(async (id: number) => {
    try { setSelected(await call<Rec>(`/v1/ledger/${id}`)); } catch (e) { setMsg(`Trade ${id}: ${(e as Error).message}`); }
  }, []);

  const afterChange = useCallback(async (id: number, note: string) => {
    setMsg(note);
    await loadLedger();
    await openRec(id);
  }, [loadLedger, openRec]);

  return (
    <>
      <section className="controls-band">
        <label className="hint" htmlFor="k4-list">Trade list</label>
        <select id="k4-list" value={chosen} onChange={(e) => setChosen(e.target.value)} style={{ minWidth: 260 }}>
          {!lists.length && <option value="">No saved lists</option>}
          {lists.map((l) => (
            <option key={`${l.date}|${l.decision_time}`} value={`${l.date}|${l.decision_time}`}>
              {l.date} · {l.decision_time} · {l.mode} · {l.new_trades} new
            </option>
          ))}
        </select>
        <button type="button" onClick={refresh} title="Reload lists and ledger"><RefreshCw size={18} /></button>
        <span className="hint" role="status">{msg}</span>
      </section>

      {doc && <ListView doc={doc} onPick={openRec} />}

      <section className="panel" style={{ marginTop: 14 }}>
        <div className="panel-title"><h2>Open and recommended trades</h2><span>{open.length} in the ledger</span></div>
        <RecTable rows={open} onPick={openRec} selectedId={selected?.id} />
      </section>

      {selected && <TradeDetail rec={selected} onChanged={afterChange} onError={setMsg} />}

      <section className="panel" style={{ marginTop: 14 }}>
        <div className="panel-title"><h2>Closed trades</h2><span>{closed.length}</span></div>
        <RecTable rows={closed} onPick={openRec} selectedId={selected?.id} showResult />
      </section>
    </>
  );
}

function ListView({ doc, onPick }: { doc: TradeListDoc; onPick: (id: number) => void }) {
  return (
    <section className="panel">
      <div className="panel-title">
        <h2>Trade list {doc.date} · {doc.decision_time}</h2>
        <span>{doc.mode} · data up to {doc.data_up_to}</span>
      </div>
      {doc.no_trade_today && <p className="hint" style={{ padding: "4px 16px 12px" }}><strong>{doc.message}</strong></p>}
      {doc.new_trades.map((t, i) => (
        <div key={i} style={{ padding: "8px 16px 14px", borderTop: i ? "1px solid var(--line, #e5e7eb)" : undefined }}>
          <h3 style={{ margin: "6px 0" }}>
            {t.instrument}: {t.structure.replace(/_/g, " ")} <span className="hint">({t.strategy})</span>
            {t.ledger_id != null && <button type="button" className="symbol-link" onClick={() => onPick(t.ledger_id!)} style={{ marginLeft: 8 }}>ledger #{t.ledger_id}</button>}
          </h3>
          <div className="metric-strip">
            <span className="metric">Lots <strong>{t.lots}</strong> (lot {t.lot_size})</span>
            <span className="metric">Max loss <strong>{rupees(t.max_loss_rupees)}</strong> ({rupees(t.max_loss_per_lot)}/lot)</span>
            <span className="metric" title={t.margin_inputs?.worst_scenario ?? t.margin_inputs?.method ?? ""}>Margin est. <strong>{rupees(t.margin_estimate_rupees)}</strong></span>
            <span className="metric">Expected result <strong>{t.expected_result_pct_of_max_loss == null ? "none yet" : `${fixed(t.expected_result_pct_of_max_loss)}%`}</strong> of max loss</span>
          </div>
          <p className="hint" style={{ margin: "6px 0" }}>Enter {t.entry.date}, {t.entry.window}, net price limit {fixed(t.entry.net_price_limit_per_share)} per share. Exit rules: {Object.entries(t.exit_rules).map(([k, v]) => `${k.replace(/_/g, " ")} ${String(v)}`).join(", ")}</p>
          <div className="table-wrap"><table>
            <thead><tr><th>Leg</th><th>Action</th><th>Expiry</th><th>Strike</th><th>Type</th><th>Reference price</th></tr></thead>
            <tbody>{t.legs.map((l, j) => <tr key={j}><td>{j}</td><td>{l.action}</td><td>{l.expiry}</td><td>{l.strike}</td><td>{l.type}</td><td>{fixed(l.reference_price)}</td></tr>)}</tbody>
          </table></div>
        </div>
      ))}
      {doc.open_trades.length > 0 && <>
        <h3 style={{ margin: "10px 16px 4px" }}>Today's instructions for open trades</h3>
        <div className="table-wrap"><table>
          <thead><tr><th>Id</th><th>Instrument</th><th>Strategy</th><th>Sessions held</th><th>Result so far %</th><th>Instruction</th></tr></thead>
          <tbody>{doc.open_trades.map((o) => <tr key={o.id}><td><button type="button" className="symbol-link" onClick={() => onPick(o.id)}>{o.id}</button></td><td>{o.instrument}</td><td>{o.strategy}</td><td>{o.sessions_held ?? "—"}</td><td>{fixed(o.result_pct_so_far, 1)}</td><td><strong>{o.instruction}</strong></td></tr>)}</tbody>
        </table></div>
      </>}
      {doc.best_refused.length > 0 && <>
        <h3 style={{ margin: "10px 16px 4px" }}>Best candidates that did not trade</h3>
        <div className="table-wrap"><table>
          <thead><tr><th>Instrument</th><th>Strategy</th><th>Expected result %</th><th>Score</th><th>Reason</th></tr></thead>
          <tbody>{doc.best_refused.map((r, i) => <tr key={i}><td>{r.instrument}</td><td>{r.strategy}</td><td>{r.expected_result_pct == null ? "none yet" : fixed(r.expected_result_pct)}</td><td>{fixed(r.score, 3)}</td><td>{r.reason.replace(/_/g, " ")}</td></tr>)}</tbody>
        </table></div>
      </>}
      {(doc.data_warnings?.length ?? 0) > 0 && <>
        <h3 style={{ margin: "10px 16px 4px" }}>Data warnings</h3>
        <ul className="hint" style={{ margin: "0 16px 12px" }}>{doc.data_warnings!.map((w, i) => <li key={i}>{w.file}: {w.detail}</li>)}</ul>
      </>}
    </section>
  );
}

function RecTable({ rows, onPick, selectedId, showResult = false }: { rows: Rec[]; onPick: (id: number) => void; selectedId?: number; showResult?: boolean }) {
  if (!rows.length) return <p className="hint" style={{ padding: "4px 16px 12px" }}>None.</p>;
  return (
    <div className="table-wrap"><table>
      <thead><tr><th>Id</th><th>Entry date</th><th>Instrument</th><th>Strategy</th><th>Status</th><th>Mode</th><th>Lots</th><th>Max loss</th>{showResult && <><th>Exit</th><th>P&amp;L</th><th>Result %</th></>}</tr></thead>
      <tbody>{rows.map((r) => (
        <tr key={r.id} className={r.id === selectedId ? "picked-row" : undefined}>
          <td><button type="button" className="symbol-link" onClick={() => onPick(r.id)}>{r.id}</button></td>
          <td>{r.entry_date}</td><td>{r.instrument}</td><td>{r.strategy}</td><td>{r.status}</td><td>{r.mode}</td><td>{r.lots}</td><td>{rupees(r.max_loss)}</td>
          {showResult && <><td>{r.exit_date ?? "—"} {r.exit_reason ? <span className="hint">({r.exit_reason})</span> : null}</td><td>{rupees(r.pnl)}</td><td>{fixed(r.result_pct, 1)}</td></>}
        </tr>
      ))}</tbody>
    </table></div>
  );
}

function TradeDetail({ rec, onChanged, onError }: { rec: Rec; onChanged: (id: number, note: string) => Promise<void>; onError: (m: string) => void }) {
  const qty = rec.lots * rec.lot_size;
  const nowLocal = () => { const d = new Date(); d.setMinutes(d.getMinutes() - d.getTimezoneOffset()); return d.toISOString().slice(0, 16); };
  const [kind, setKind] = useState<"entry" | "exit">("entry");
  const [leg, setLeg] = useState(0);
  const [price, setPrice] = useState("");
  const [quantity, setQuantity] = useState(String(qty));
  const [filledAt, setFilledAt] = useState(nowLocal);
  const [note, setNote] = useState("");
  const [exitDate, setExitDate] = useState(new Date().toLocaleDateString("en-CA"));
  const [exitReason, setExitReason] = useState("");
  const [busy, setBusy] = useState(false);

  const filled = useMemo(() => {
    const has = (k: string, i: number) => (rec.fills ?? []).some((f) => f.kind === k && f.leg === i);
    return rec.legs.map((_, i) => ({ entry: has("entry", i), exit: has("exit", i) }));
  }, [rec]);

  // Point the form at the next fill still missing: every leg's entry first, then every leg's exit.
  useEffect(() => {
    const entryLeg = filled.findIndex((f) => !f.entry);
    const exitLeg = filled.findIndex((f) => !f.exit);
    if (entryLeg >= 0) { setKind("entry"); setLeg(entryLeg); }
    else { setKind("exit"); setLeg(Math.max(exitLeg, 0)); }
    setPrice("");
    setQuantity(String(rec.lots * rec.lot_size));
  }, [filled, rec.lots, rec.lot_size]);
  const canClose = rec.status === "open" && filled.every((f) => f.entry && f.exit);
  const editable = rec.status === "recommended" || rec.status === "open";

  async function act(fn: () => Promise<unknown>, note: string) {
    setBusy(true);
    try { await fn(); await onChanged(rec.id, note); } catch (e) { onError(`Trade ${rec.id}: ${(e as Error).message}`); } finally { setBusy(false); }
  }
  const submitFill = (e: React.FormEvent) => {
    e.preventDefault();
    const p = Number(price), q = Number(quantity);
    if (!(p >= 0) || price === "" || !(q > 0)) { onError("A fill needs a price of 0 or more and a quantity above 0."); return; }
    act(() => call("/v1/ledger/fills", { rec_id: rec.id, kind, leg, price: p, quantity: q, filled_at: filledAt.replace("T", " "), note }),
      `Recorded ${kind} fill for trade ${rec.id}, leg ${leg}.`).then(() => { setPrice(""); setNote(""); });
  };

  return (
    <section className="panel" style={{ marginTop: 14 }}>
      <div className="panel-title"><h2>Trade #{rec.id}: {rec.instrument} · {rec.strategy}</h2><span>{rec.status} · {rec.mode} · {rec.lots} lots × {rec.lot_size}</span></div>
      <div className="table-wrap"><table>
        <thead><tr><th>Leg</th><th>Contract</th><th>Reference price</th><th>Entry fill</th><th>Exit fill</th></tr></thead>
        <tbody>{rec.legs.map((l, i) => {
          const fs = (k: string) => (rec.fills ?? []).filter((f) => f.kind === k && f.leg === i).map((f) => `${fixed(f.price)} × ${f.quantity}`).join(", ") || "—";
          return <tr key={i}><td>{i}</td><td>{legLabel(l)}</td><td>{fixed(rec.reference_prices?.[i])}</td><td>{fs("entry")}</td><td>{fs("exit")}</td></tr>;
        })}</tbody>
      </table></div>

      {editable && (
        <form onSubmit={submitFill} className="controls-band" style={{ flexWrap: "wrap" }}>
          <label className="hint" htmlFor="k4-kind">Fill</label>
          <select id="k4-kind" value={kind} onChange={(e) => setKind(e.target.value as "entry" | "exit")}>
            <option value="entry">entry</option><option value="exit">exit</option>
          </select>
          <label className="hint" htmlFor="k4-leg">Leg</label>
          <select id="k4-leg" value={leg} onChange={(e) => setLeg(Number(e.target.value))}>
            {rec.legs.map((l, i) => <option key={i} value={i}>{i}: {legLabel(l)}</option>)}
          </select>
          <label className="hint" htmlFor="k4-price">Price</label>
          <input id="k4-price" type="number" step="0.05" min="0" value={price} onChange={(e) => setPrice(e.target.value)} required style={{ width: 100 }} />
          <label className="hint" htmlFor="k4-qty">Quantity</label>
          <input id="k4-qty" type="number" step="1" min="1" value={quantity} onChange={(e) => setQuantity(e.target.value)} required style={{ width: 100 }} />
          <label className="hint" htmlFor="k4-at">Filled at</label>
          <input id="k4-at" type="datetime-local" value={filledAt} onChange={(e) => setFilledAt(e.target.value)} required />
          <input aria-label="Note" placeholder="note" value={note} onChange={(e) => setNote(e.target.value)} style={{ width: 160 }} />
          <button type="submit" className="btn-wide" disabled={busy}>Record fill</button>
        </form>
      )}

      {rec.status === "recommended" && (
        <div className="controls-band">
          <button type="button" disabled={busy} onClick={() => act(() => call(`/v1/ledger/${rec.id}/not-taken`, { note }), `Trade ${rec.id} marked not taken.`)}>Mark not taken</button>
          <span className="hint">Only before any fill. The note box above is saved with it.</span>
        </div>
      )}

      {rec.status === "open" && (
        <div className="controls-band">
          <label className="hint" htmlFor="k4-exit-date">Exit date</label>
          <input id="k4-exit-date" type="date" value={exitDate} onChange={(e) => setExitDate(e.target.value)} />
          <input aria-label="Exit reason" placeholder="exit reason (e.g. target, max hold)" value={exitReason} onChange={(e) => setExitReason(e.target.value)} style={{ width: 240 }} />
          <button type="button" style={{ whiteSpace: "nowrap" }} disabled={busy || !canClose || !exitReason} onClick={() => act(() => call(`/v1/ledger/${rec.id}/close`, { exit_date: exitDate, exit_reason: exitReason }), `Trade ${rec.id} closed.`)}>Close trade</button>
          {!canClose && <span className="hint">Every leg needs an entry and an exit fill first.</span>}
        </div>
      )}

      {rec.status === "closed" && <p className="hint" style={{ padding: "4px 16px 12px" }}>Closed {rec.exit_date} ({rec.exit_reason}): P&amp;L {rupees(rec.pnl)}, {fixed(rec.result_pct, 1)}% of max loss.</p>}
    </section>
  );
}
