import React, { useEffect, useState, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from "recharts";

const API = "https://f1-ai-platform-production.up.railway.app";


const COMPOUND_COLORS = {
  SOFT: "#e10600", MEDIUM: "#ffd700", HARD: "#eeeeee",
  INTERMEDIATE: "#39b54a", WET: "#0067ff",
};

const DRIVER_COLORS = [
  "#e10600", "#00d2be", "#ff8700", "#dc0000",
  "#0600ef", "#005aff", "#900000", "#ffffff",
  "#2293d1", "#37bedd", "#b6babd", "#c92d4b",
  "#006f62", "#0090ff", "#1e6176", "#f596c8",
  "#358c75", "#fd4bc7", "#d3ab17", "#aaaaaa",
];

function formatLapTime(seconds) {
  if (!seconds && seconds !== 0) return "—";
  const mins = Math.floor(seconds / 60);
  const secs = (seconds % 60).toFixed(3).padStart(6, "0");
  return `${mins}:${secs}`;
}

function formatRaceTime(seconds) {
  if (!seconds) return "—";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = (seconds % 60).toFixed(1);
  return h > 0 ? `${h}h ${m}m ${s}s` : `${m}m ${s}s`;
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "#1a1a1a", border: "1px solid #444", borderRadius: 6, padding: "10px 14px", fontSize: 13 }}>
      <div style={{ color: "#aaa", marginBottom: 6 }}>Lap {label}</div>
      {payload.map((p) => (
        <div key={p.name} style={{ color: p.color, marginBottom: 2 }}>
          {p.name}: <strong>{formatLapTime(p.value)}</strong>
        </div>
      ))}
    </div>
  );
}

function CompoundBadge({ compound }) {
  const bg = COMPOUND_COLORS[compound] ?? "#555";
  const textColor = (compound === "HARD" || compound === "MEDIUM") ? "#000" : "#fff";
  return (
    <span style={{ background: bg, color: textColor, fontSize: 10, fontWeight: 700, padding: "2px 7px", borderRadius: 4, letterSpacing: 0.5 }}>
      {compound ?? "—"}
    </span>
  );
}

function DnfBadge() {
  return <sup style={{ fontSize: 9, color: "#ff5555", fontWeight: 700, letterSpacing: 0.5, marginLeft: 1 }}>DNF</sup>;
}

function Podium({ podium }) {
  if (!podium || podium.length < 3) return null;
  const p1 = podium.find((p) => p.ClassifiedPosition === 1);
  const p2 = podium.find((p) => p.ClassifiedPosition === 2);
  const p3 = podium.find((p) => p.ClassifiedPosition === 3);
  const MEDAL = { 1: "#ffd700", 2: "#c0c0c0", 3: "#cd7f32" };
  function Stand({ driver, pos, height }) {
    return (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
        <div style={{ fontSize: 15, fontWeight: 700, color: MEDAL[pos], letterSpacing: 1 }}>{driver}</div>
        <div style={{ fontSize: 11, color: "#666", fontWeight: 600, marginTop: -4 }}>P{pos}</div>
        <div style={{ width: 72, height, background: `${MEDAL[pos]}18`, border: `1px solid ${MEDAL[pos]}55`, borderRadius: "6px 6px 0 0", display: "flex", alignItems: "flex-start", justifyContent: "center", paddingTop: 10 }}>
          <span style={{ fontSize: 22 }}>{pos === 1 ? "🏆" : pos === 2 ? "🥈" : "🥉"}</span>
        </div>
      </div>
    );
  }
  return (
    <div style={{ background: "#1a1a1a", border: "1px solid #333", borderRadius: 8, padding: "20px 28px 0", display: "flex", flexDirection: "column", alignItems: "center", minWidth: 260 }}>
      <div style={{ color: "#888", fontSize: 11, textTransform: "uppercase", letterSpacing: 1, marginBottom: 16 }}>Podium</div>
      <div style={{ display: "flex", alignItems: "flex-end", gap: 4 }}>
        <Stand driver={p2?.Abbreviation} pos={2} height={80} />
        <Stand driver={p1?.Abbreviation} pos={1} height={110} />
        <Stand driver={p3?.Abbreviation} pos={3} height={56} />
      </div>
    </div>
  );
}

function FastestLapCard({ fastestLap }) {
  if (!fastestLap) return null;
  return (
    <div style={{ background: "#1a1a1a", border: "1px solid #333", borderRadius: 8, padding: "14px 20px", minWidth: 200 }}>
      <div style={{ color: "#888", fontSize: 11, textTransform: "uppercase", letterSpacing: 1 }}>Fastest Lap of Race</div>
      <div style={{ color: "#e10600", fontSize: 26, fontWeight: 700, marginTop: 6, fontFamily: "monospace" }}>{formatLapTime(fastestLap.timeSeconds)}</div>
      <div style={{ color: "#aaa", fontSize: 13, marginTop: 4 }}>
        {fastestLap.driver}<span style={{ color: "#555", marginLeft: 8 }}>· Lap {fastestLap.lapNumber}</span>
      </div>
      <div style={{ display: "inline-block", marginTop: 8, background: "#e1060018", border: "1px solid #e1060044", borderRadius: 4, padding: "2px 8px", fontSize: 10, color: "#e10600", fontWeight: 700, letterSpacing: 1 }}>
        ⚡ FASTEST LAP
      </div>
    </div>
  );
}

function StintChart({ stints, allDrivers, driverStatus }) {
  const maxLap = Math.max(...stints.map((s) => s.EndLap), 1);
  const driversWithStints = allDrivers.filter((d) => stints.some((s) => s.Driver === d));
  return (
    <div style={{ overflowX: "auto" }}>
      {driversWithStints.map((driver) => {
        const driverStints = stints.filter((s) => s.Driver === driver);
        const dnf = driverStatus[driver] && !driverStatus[driver].finished;
        return (
          <div key={driver} style={{ display: "flex", alignItems: "center", marginBottom: 6, gap: 10 }}>
            <div style={{ width: 40, fontSize: 11, fontWeight: 700, color: dnf ? "#ff5555" : "#aaa", textAlign: "right", flexShrink: 0 }}>
              {driver}{dnf && <sup style={{ fontSize: 8, color: "#ff5555" }}>DNF</sup>}
            </div>
            <div style={{ flex: 1, height: 22, background: "#1a1a1a", borderRadius: 4, position: "relative", minWidth: 0 }}>
              {driverStints.map((stint, i) => {
                const left = ((stint.StartLap - 1) / maxLap) * 100;
                const width = ((stint.EndLap - stint.StartLap + 1) / maxLap) * 100;
                const color = COMPOUND_COLORS[stint.Compound] ?? "#555";
                const textColor = (stint.Compound === "HARD" || stint.Compound === "MEDIUM") ? "#000" : "#fff";
                return (
                  <div key={i} title={`${stint.Compound} · Laps ${stint.StartLap}–${stint.EndLap}`}
                    style={{ position: "absolute", left: `${left}%`, width: `${width}%`, height: "100%", background: color, borderRadius: 3, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, fontWeight: 700, color: textColor, overflow: "hidden", boxSizing: "border-box", borderRight: "2px solid #0f0f0f" }}>
                    {width > 8 ? stint.Compound[0] : ""}
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 8 }}>
        <div style={{ width: 40 }} />
        <div style={{ flex: 1, display: "flex", justifyContent: "space-between" }}>
          {[1, Math.round(maxLap * 0.25), Math.round(maxLap * 0.5), Math.round(maxLap * 0.75), maxLap].map((lap) => (
            <span key={lap} style={{ fontSize: 10, color: "#555" }}>Lap {lap}</span>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Lap Time Predictor ───────────────────────────────────────────────────────
function LapTimePredictor({ laps, modelInfo }) {
  const [compound, setCompound] = useState("MEDIUM");
  const [tyreLife, setTyreLife] = useState(15);
  const [lapNumber, setLapNumber] = useState(30);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const baseline = React.useMemo(() => {
    const times = laps.map((l) => l.LapTimeSeconds).filter(Boolean).sort((a, b) => a - b);
    if (!times.length) return 0;
    return times[Math.floor(times.length * 0.05)];
  }, [laps]);

  const totalLaps = React.useMemo(() => {
    const lapNums = laps.map((l) => l.LapNumber).filter(Boolean);
    return lapNums.length ? Math.max(...lapNums) : 78;
  }, [laps]);

  async function predict() {
    setLoading(true);
    try {
      const params = new URLSearchParams({ compound, tyre_life: tyreLife, lap_number: lapNumber, total_laps: totalLaps, baseline_seconds: baseline });
      const res = await fetch(`${API}/predict-laptime?${params}`);
      const data = await res.json();
      setPrediction(data);
    } catch (e) { console.error(e); }
    setLoading(false);
  }

  const SLIDER_STYLE = { width: "100%", accentColor: "#e10600", cursor: "pointer" };
  const INPUT_LABEL = { color: "#888", fontSize: 11, textTransform: "uppercase", letterSpacing: 1, marginBottom: 8, display: "block" };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
      <div style={{ background: "#141414", border: "1px solid #222", borderRadius: 10, padding: 24 }}>
        <div style={{ background: "#0f0f0f", border: "1px solid #333", borderRadius: 6, padding: "10px 14px", marginBottom: 24, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <div style={{ color: "#666", fontSize: 10, textTransform: "uppercase", letterSpacing: 1 }}>XGBoost Model</div>
            <div style={{ color: "#aaa", fontSize: 12, marginTop: 2 }}>Trained on {modelInfo?.trainedOnLaps?.toLocaleString() ?? "—"} laps</div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div style={{ color: "#666", fontSize: 10, textTransform: "uppercase", letterSpacing: 1 }}>Accuracy</div>
            <div style={{ color: "#00d2be", fontSize: 14, fontWeight: 700, marginTop: 2 }}>±{modelInfo?.maeSeconds?.toFixed(2) ?? "—"}s MAE</div>
          </div>
        </div>
        <div style={{ marginBottom: 20 }}>
          <span style={INPUT_LABEL}>Tyre Compound</span>
          <div style={{ display: "flex", gap: 8 }}>
            {["SOFT", "MEDIUM", "HARD"].map((c) => (
              <button key={c} onClick={() => setCompound(c)} style={{ padding: "7px 16px", borderRadius: 6, cursor: "pointer", fontWeight: 700, fontSize: 12, letterSpacing: 0.5, border: `2px solid ${compound === c ? COMPOUND_COLORS[c] : "#333"}`, background: compound === c ? COMPOUND_COLORS[c] + "22" : "transparent", color: compound === c ? COMPOUND_COLORS[c] : "#555", transition: "all 0.15s" }}>{c}</button>
            ))}
          </div>
        </div>
        <div style={{ marginBottom: 20 }}>
          <span style={INPUT_LABEL}>Tyre Age — <span style={{ color: "#fff" }}>{tyreLife} laps</span></span>
          <input type="range" min={1} max={50} value={tyreLife} onChange={(e) => setTyreLife(Number(e.target.value))} style={SLIDER_STYLE} />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#555", marginTop: 4 }}><span>Fresh (1)</span><span>Old (50)</span></div>
        </div>
        <div style={{ marginBottom: 24 }}>
          <span style={INPUT_LABEL}>Race Lap — <span style={{ color: "#fff" }}>Lap {lapNumber}</span><span style={{ color: "#555", marginLeft: 8 }}>of {totalLaps}</span></span>
          <input type="range" min={1} max={totalLaps} value={lapNumber} onChange={(e) => setLapNumber(Number(e.target.value))} style={SLIDER_STYLE} />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#555", marginTop: 4 }}><span>Lap 1</span><span>Lap {totalLaps}</span></div>
        </div>
        <button onClick={predict} disabled={loading || !baseline} style={{ width: "100%", padding: "12px", background: loading ? "#333" : "#e10600", border: "none", borderRadius: 8, color: "#fff", fontSize: 14, fontWeight: 700, cursor: loading ? "not-allowed" : "pointer", letterSpacing: 0.5, transition: "background 0.2s" }}>
          {loading ? "Predicting..." : "⚡ Predict Lap Time"}
        </button>
      </div>
      <div style={{ background: "#141414", border: "1px solid #222", borderRadius: 10, padding: 24, display: "flex", flexDirection: "column", gap: 20 }}>
        {prediction ? (
          <>
            <div style={{ textAlign: "center", padding: "20px 0" }}>
              <div style={{ color: "#666", fontSize: 11, textTransform: "uppercase", letterSpacing: 2, marginBottom: 8 }}>Predicted Lap Time</div>
              <div style={{ color: "#e10600", fontSize: 48, fontWeight: 700, fontFamily: "monospace", letterSpacing: -1 }}>{formatLapTime(prediction.predictedSeconds)}</div>
              <div style={{ color: "#555", fontSize: 13, marginTop: 8 }}>+{prediction.predictedDelta?.toFixed(3)}s above race baseline</div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
              {[{ label: "Compound", value: prediction.compound }, { label: "Tyre Age", value: `${prediction.tyreLife} laps` }, { label: "Race Lap", value: `Lap ${prediction.lapNumber}` }].map(({ label, value }) => (
                <div key={label} style={{ background: "#0f0f0f", borderRadius: 6, padding: "10px 12px", textAlign: "center" }}>
                  <div style={{ color: "#555", fontSize: 10, textTransform: "uppercase", letterSpacing: 1 }}>{label}</div>
                  <div style={{ color: "#ccc", fontSize: 13, fontWeight: 600, marginTop: 4 }}>{value}</div>
                </div>
              ))}
            </div>
            <div style={{ background: "#00d2be11", border: "1px solid #00d2be33", borderRadius: 6, padding: "10px 14px", fontSize: 12, color: "#00d2be" }}>
              Model accuracy: ±{prediction.modelMAE?.toFixed(2)}s — prediction is within this margin ~68% of the time
            </div>
            <div style={{ background: "#0f0f0f", borderRadius: 6, padding: "12px 14px" }}>
              <div style={{ color: "#666", fontSize: 10, textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 }}>Tyre Degradation Insight</div>
              <div style={{ color: "#aaa", fontSize: 13, lineHeight: 1.6 }}>
                {prediction.compound === "SOFT" && "Soft tyres degrade fastest — expect significant time loss after lap 15."}
                {prediction.compound === "MEDIUM" && "Medium tyres offer the best balance — degradation is gradual and manageable."}
                {prediction.compound === "HARD" && "Hard tyres are the most durable — minimal degradation but slower initial pace."}
              </div>
            </div>
          </>
        ) : (
          <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", color: "#333", textAlign: "center" }}>
            <div style={{ fontSize: 48, marginBottom: 12 }}>🔮</div>
            <div style={{ fontSize: 14, marginBottom: 6 }}>Configure inputs and click predict</div>
            <div style={{ fontSize: 12 }}>Uses XGBoost trained on 11,000+ F1 laps</div>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Strategy Simulator ───────────────────────────────────────────────────────
function StrategySimulator({ allDrivers, driverStatus, selectedYear, selectedGp }) {
  const [driver, setDriver] = useState("");
  const [pitLap, setPitLap] = useState(30);
  const [compound1, setCompound1] = useState("MEDIUM");
  const [compound2, setCompound2] = useState("HARD");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // Default to first driver with lap data
  useEffect(() => {
    if (allDrivers.length && !driver) setDriver(allDrivers[0]);
  }, [allDrivers, driver]);

  async function simulate() {
    if (!driver) return;
    setLoading(true);
    setResult(null);
    try {
      const params = new URLSearchParams({
        year: selectedYear, gp: selectedGp,
        driver, custom_pit_lap: pitLap,
        custom_compound_1: compound1,
        custom_compound_2: compound2,
      });
      const res = await fetch(`${API}/simulate-strategy?${params}`);
      const data = await res.json();
      setResult(data);
      // Set pit lap slider max to total laps
      if (data.totalLaps && pitLap > data.totalLaps - 2) {
        setPitLap(Math.floor(data.totalLaps / 2));
      }
    } catch (e) { console.error(e); }
    setLoading(false);
  }

  const SLIDER_STYLE = { width: "100%", accentColor: "#00d2be", cursor: "pointer" };
  const INPUT_LABEL = { color: "#888", fontSize: 11, textTransform: "uppercase", letterSpacing: 1, marginBottom: 8, display: "block" };

  const totalLaps = result?.totalLaps ?? 78;
  const delta = result?.delta;
  const fasterByCustom = delta !== null && delta < 0;
  const deltaAbs = delta !== null ? Math.abs(delta) : null;

  // Build comparison chart data
  const comparisonChartData = React.useMemo(() => {
    if (!result?.actualLaps?.length) return [];
    const actualMap = {};
    result.actualLaps.forEach((l) => { actualMap[l.lap] = l.predictedSeconds; });
    const customMap = {};
    result.customLaps?.forEach((l) => { customMap[l.lap] = l.predictedSeconds; });
    const laps = result.actualLaps.map((l) => ({
      lap: l.lap,
      Actual: l.predictedSeconds,
      Custom: customMap[l.lap] ?? null,
    }));
    return laps;
  }, [result]);

  // Build stint bar for actual strategy
  function ActualStintBar() {
    if (!result?.actualStints?.length) return null;
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
        <span style={{ color: "#666", fontSize: 11, width: 50, flexShrink: 0 }}>Actual</span>
        <div style={{ flex: 1, height: 24, position: "relative", background: "#1a1a1a", borderRadius: 4 }}>
          {result.actualStints.map((stint, i) => {
            const left = ((stint.StartLap - 1) / totalLaps) * 100;
            const width = ((stint.EndLap - stint.StartLap + 1) / totalLaps) * 100;
            const color = COMPOUND_COLORS[stint.Compound] ?? "#555";
            const textColor = (stint.Compound === "HARD" || stint.Compound === "MEDIUM") ? "#000" : "#fff";
            return (
              <div key={i} style={{ position: "absolute", left: `${left}%`, width: `${width}%`, height: "100%", background: color, borderRadius: 3, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, fontWeight: 700, color: textColor, overflow: "hidden", borderRight: "2px solid #0f0f0f", boxSizing: "border-box" }}>
                {width > 8 ? stint.Compound[0] : ""}
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  function CustomStintBar() {
    const width1 = (pitLap / totalLaps) * 100;
    const width2 = ((totalLaps - pitLap) / totalLaps) * 100;
    const color1 = COMPOUND_COLORS[compound1] ?? "#555";
    const color2 = COMPOUND_COLORS[compound2] ?? "#555";
    const text1 = (compound1 === "HARD" || compound1 === "MEDIUM") ? "#000" : "#fff";
    const text2 = (compound2 === "HARD" || compound2 === "MEDIUM") ? "#000" : "#fff";
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{ color: "#00d2be", fontSize: 11, width: 50, flexShrink: 0 }}>Custom</span>
        <div style={{ flex: 1, height: 24, position: "relative", background: "#1a1a1a", borderRadius: 4 }}>
          <div style={{ position: "absolute", left: "0%", width: `${width1}%`, height: "100%", background: color1, borderRadius: 3, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, fontWeight: 700, color: text1, overflow: "hidden", borderRight: "2px solid #0f0f0f", boxSizing: "border-box" }}>
            {width1 > 8 ? compound1[0] : ""}
          </div>
          <div style={{ position: "absolute", left: `${width1}%`, width: `${width2}%`, height: "100%", background: color2, borderRadius: 3, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, fontWeight: 700, color: text2, overflow: "hidden", boxSizing: "border-box" }}>
            {width2 > 8 ? compound2[0] : ""}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ background: "#141414", border: "1px solid #222", borderRadius: 10, padding: 24 }}>

      {/* Controls row */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr auto", gap: 20, marginBottom: 28, alignItems: "flex-end" }}>

        {/* Driver picker */}
        <div>
          <span style={INPUT_LABEL}>Driver</span>
          <select value={driver} onChange={(e) => setDriver(e.target.value)}
            style={{ background: "#0f0f0f", border: "1px solid #333", borderRadius: 6, color: "#fff", padding: "8px 12px", fontSize: 13, cursor: "pointer", outline: "none", width: "100%" }}>
            {allDrivers.filter((d) => !driverStatus[d] || driverStatus[d].finished).map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </div>

        {/* Compound 1 */}
        <div>
          <span style={INPUT_LABEL}>Stint 1 Compound</span>
          <div style={{ display: "flex", gap: 6 }}>
            {["SOFT", "MEDIUM", "HARD"].map((c) => (
              <button key={c} onClick={() => setCompound1(c)} style={{ flex: 1, padding: "6px 4px", borderRadius: 5, cursor: "pointer", fontWeight: 700, fontSize: 10, border: `2px solid ${compound1 === c ? COMPOUND_COLORS[c] : "#333"}`, background: compound1 === c ? COMPOUND_COLORS[c] + "22" : "transparent", color: compound1 === c ? COMPOUND_COLORS[c] : "#555" }}>
                {c[0]}
              </button>
            ))}
          </div>
        </div>

        {/* Compound 2 */}
        <div>
          <span style={INPUT_LABEL}>Stint 2 Compound</span>
          <div style={{ display: "flex", gap: 6 }}>
            {["SOFT", "MEDIUM", "HARD"].map((c) => (
              <button key={c} onClick={() => setCompound2(c)} style={{ flex: 1, padding: "6px 4px", borderRadius: 5, cursor: "pointer", fontWeight: 700, fontSize: 10, border: `2px solid ${compound2 === c ? COMPOUND_COLORS[c] : "#333"}`, background: compound2 === c ? COMPOUND_COLORS[c] + "22" : "transparent", color: compound2 === c ? COMPOUND_COLORS[c] : "#555" }}>
                {c[0]}
              </button>
            ))}
          </div>
        </div>

        {/* Pit lap */}
        <div>
          <span style={INPUT_LABEL}>Pit on Lap — <span style={{ color: "#fff" }}>{pitLap}</span></span>
          <input type="range" min={1} max={totalLaps - 1} value={pitLap}
            onChange={(e) => setPitLap(Number(e.target.value))} style={SLIDER_STYLE} />
        </div>

        {/* Run button */}
        <button onClick={simulate} disabled={loading}
          style={{ padding: "10px 20px", background: loading ? "#333" : "#00d2be", border: "none", borderRadius: 8, color: "#000", fontWeight: 700, fontSize: 13, cursor: loading ? "not-allowed" : "pointer", whiteSpace: "nowrap" }}>
          {loading ? "Running..." : "▶ Simulate"}
        </button>
      </div>

      {result && !result.error && (
        <>
          {/* Strategy bars */}
          <div style={{ marginBottom: 24 }}>
            <div style={{ color: "#666", fontSize: 10, textTransform: "uppercase", letterSpacing: 1, marginBottom: 12 }}>Strategy Comparison</div>
            <ActualStintBar />
            <CustomStintBar />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#555", marginTop: 6, paddingLeft: 60 }}>
              <span>Lap 1</span>
              <span>Lap {Math.round(totalLaps * 0.25)}</span>
              <span>Lap {Math.round(totalLaps * 0.5)}</span>
              <span>Lap {Math.round(totalLaps * 0.75)}</span>
              <span>Lap {totalLaps}</span>
            </div>
          </div>

          {/* Result cards */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 24 }}>
            <div style={{ background: "#0f0f0f", borderRadius: 8, padding: "14px 16px" }}>
              <div style={{ color: "#666", fontSize: 10, textTransform: "uppercase", letterSpacing: 1 }}>Actual Strategy</div>
              <div style={{ color: "#aaa", fontSize: 18, fontWeight: 700, fontFamily: "monospace", marginTop: 6 }}>{formatRaceTime(result.actualPredictedTotal)}</div>
              <div style={{ color: "#555", fontSize: 11, marginTop: 4 }}>Predicted race time</div>
            </div>
            <div style={{ background: "#0f0f0f", borderRadius: 8, padding: "14px 16px" }}>
              <div style={{ color: "#00d2be", fontSize: 10, textTransform: "uppercase", letterSpacing: 1 }}>Custom Strategy</div>
              <div style={{ color: "#00d2be", fontSize: 18, fontWeight: 700, fontFamily: "monospace", marginTop: 6 }}>{formatRaceTime(result.customPredictedTotal)}</div>
              <div style={{ color: "#555", fontSize: 11, marginTop: 4 }}>Predicted race time</div>
            </div>
            <div style={{ background: fasterByCustom ? "#00d2be11" : "#e1060011", border: `1px solid ${fasterByCustom ? "#00d2be33" : "#e1060033"}`, borderRadius: 8, padding: "14px 16px" }}>
              <div style={{ color: fasterByCustom ? "#00d2be" : "#e10600", fontSize: 10, textTransform: "uppercase", letterSpacing: 1 }}>
                {fasterByCustom ? "✓ Custom is Faster" : "✗ Custom is Slower"}
              </div>
              <div style={{ color: fasterByCustom ? "#00d2be" : "#e10600", fontSize: 24, fontWeight: 700, fontFamily: "monospace", marginTop: 6 }}>
                {fasterByCustom ? "-" : "+"}{deltaAbs?.toFixed(1)}s
              </div>
              <div style={{ color: "#555", fontSize: 11, marginTop: 4 }}>vs actual strategy</div>
            </div>
          </div>

          {/* Lap time comparison chart */}
          <div style={{ color: "#666", fontSize: 10, textTransform: "uppercase", letterSpacing: 1, marginBottom: 12 }}>Predicted Lap Times — Actual vs Custom</div>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={comparisonChartData} margin={{ top: 5, right: 20, bottom: 10, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
              <XAxis dataKey="lap" stroke="#555" tick={{ fill: "#555", fontSize: 11 }}
                label={{ value: "Lap", position: "insideBottom", offset: -5, fill: "#555", fontSize: 11 }} />
              <YAxis stroke="#555" tick={{ fill: "#555", fontSize: 11 }} domain={["auto", "auto"]} tickFormatter={formatLapTime} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ color: "#888", fontSize: 12, paddingTop: 8 }} />
              {result.actualPitLap && (
                <ReferenceLine x={result.actualPitLap} stroke="#555" strokeDasharray="4 4" label={{ value: "Actual pit", fill: "#555", fontSize: 10 }} />
              )}
              <ReferenceLine x={pitLap} stroke="#00d2be" strokeDasharray="4 4" label={{ value: "Custom pit", fill: "#00d2be", fontSize: 10 }} />
              <Line type="monotone" dataKey="Actual" stroke="#888" dot={false} strokeWidth={1.5} />
              <Line type="monotone" dataKey="Custom" stroke="#00d2be" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </>
      )}

      {result?.error && (
        <div style={{ color: "#ff6666", fontSize: 13, padding: 16 }}>Error: {result.error}</div>
      )}

      {!result && !loading && (
        <div style={{ textAlign: "center", color: "#333", padding: "40px 0" }}>
          <div style={{ fontSize: 36, marginBottom: 12 }}>🏁</div>
          <div style={{ fontSize: 14 }}>Select a driver and strategy, then click Simulate</div>
        </div>
      )}
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [availableRaces, setAvailableRaces] = useState({});
  const [selectedYear, setSelectedYear] = useState("2024");
  const [selectedGp, setSelectedGp] = useState("Monaco Grand Prix");
  const [racesLoading, setRacesLoading] = useState(true);

  const [laps, setLaps] = useState([]);
  const [driverStatus, setDriverStatus] = useState({});
  const [stints, setStints] = useState([]);
  const [positions, setPositions] = useState([]);
  const [raceOverview, setRaceOverview] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [dataLoading, setDataLoading] = useState(false);
  const [error, setError] = useState(null);

  const [selectedDrivers, setSelectedDrivers] = useState([]);
  const [tableDriver, setTableDriver] = useState(null);
  const [allDrivers, setAllDrivers] = useState([]);

  useEffect(() => {
    fetch(`${API}/available-races`).then((r) => r.json()).then((data) => {
      setAvailableRaces(data);
      setRacesLoading(false);
      if (data["2024"]?.includes("Monaco Grand Prix")) { setSelectedYear("2024"); setSelectedGp("Monaco Grand Prix"); }
      else { const y = Object.keys(data).sort()[0]; if (y) { setSelectedYear(y); setSelectedGp(data[y][0]); } }
    }).catch((err) => { setError(err.message); setRacesLoading(false); });
    fetch(`${API}/model-info`).then((r) => r.json()).then(setModelInfo).catch(() => { });
  }, []);

  const fetchRaceData = useCallback(() => {
    if (!selectedYear || !selectedGp) return;
    setDataLoading(true); setError(null);
    const params = `year=${selectedYear}&gp=${encodeURIComponent(selectedGp)}`;
    Promise.all([
      fetch(`${API}/race-laps?${params}`).then((r) => r.json()),
      fetch(`${API}/driver-status?${params}`).then((r) => r.json()),
      fetch(`${API}/stint-data?${params}`).then((r) => r.json()),
      fetch(`${API}/position-data?${params}`).then((r) => r.json()),
      fetch(`${API}/race-overview?${params}`).then((r) => r.json()),
    ]).then(([lapData, statusData, stintData, positionData, overviewData]) => {
      setLaps(lapData); setDriverStatus(statusData); setStints(stintData);
      setPositions(positionData); setRaceOverview(overviewData);
      const drivers = Object.keys(statusData).sort();
      setAllDrivers(drivers);
      const driversWithLaps = [...new Set(lapData.map((d) => d.Driver))];
      setSelectedDrivers(driversWithLaps.slice(0, 3));
      setTableDriver(driversWithLaps[0]);
      setDataLoading(false);
    }).catch((err) => { setError(err.message); setDataLoading(false); });
  }, [selectedYear, selectedGp]);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { if (!racesLoading) fetchRaceData(); }, [selectedYear, selectedGp, racesLoading]);

  function toggleDriver(driver) {
    setSelectedDrivers((prev) => prev.includes(driver) ? prev.filter((d) => d !== driver) : [...prev, driver]);
  }

  const tableLaps = laps.filter((l) => l.Driver === tableDriver);
  const validTimes = tableLaps.map((l) => l.LapTimeSeconds).filter(Boolean);
  const fastestLap = validTimes.length ? Math.min(...validTimes) : null;
  const avgLap = validTimes.length ? validTimes.reduce((a, b) => a + b, 0) / validTimes.length : null;

  const chartData = React.useMemo(() => {
    if (!selectedDrivers.length) return [];
    const lapMap = {};
    laps.filter((l) => selectedDrivers.includes(l.Driver)).forEach((l) => {
      const ln = l.LapNumber;
      if (!lapMap[ln]) lapMap[ln] = { LapNumber: ln };
      lapMap[ln][l.Driver] = l.LapTimeSeconds;
    });
    return Object.values(lapMap).sort((a, b) => a.LapNumber - b.LapNumber);
  }, [laps, selectedDrivers]);

  const positionChartData = React.useMemo(() => {
    if (!selectedDrivers.length) return [];
    const lapMap = {};
    positions.filter((p) => selectedDrivers.includes(p.Driver)).forEach((p) => {
      const ln = p.LapNumber;
      if (!lapMap[ln]) lapMap[ln] = { LapNumber: ln };
      lapMap[ln][p.Driver] = p.Position;
    });
    return Object.values(lapMap).sort((a, b) => a.LapNumber - b.LapNumber);
  }, [positions, selectedDrivers]);

  const SELECT_STYLE = { background: "#1a1a1a", border: "1px solid #444", borderRadius: 6, color: "#fff", padding: "8px 12px", fontSize: 13, cursor: "pointer", outline: "none", minWidth: 160 };

  const styles = {
    app: { minHeight: "100vh", background: "#0f0f0f", color: "#fff", fontFamily: "'Inter', 'Helvetica Neue', sans-serif", paddingBottom: 60 },
    header: { background: "#111", borderBottom: "3px solid #e10600", padding: "18px 32px", display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 16 },
    redBar: { width: 5, height: 36, background: "#e10600", borderRadius: 2 },
    section: { maxWidth: 1200, margin: "0 auto", padding: "32px 32px 0" },
    sectionTitle: { color: "#e10600", fontSize: 11, fontWeight: 700, letterSpacing: 2, textTransform: "uppercase", marginBottom: 16 },
    driverGrid: { display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 24 },
    driverBtn: (active, color) => ({ padding: "6px 14px", border: `2px solid ${active ? color : "#333"}`, borderRadius: 20, background: active ? color + "22" : "transparent", color: active ? color : "#888", cursor: "pointer", fontSize: 13, fontWeight: 600, transition: "all 0.15s", display: "inline-flex", alignItems: "baseline", gap: 3 }),
    chartBox: { background: "#141414", border: "1px solid #222", borderRadius: 10, padding: "24px 12px 12px", marginBottom: 40 },
    table: { width: "100%", borderCollapse: "collapse", fontSize: 13 },
    th: { color: "#888", fontWeight: 600, fontSize: 11, letterSpacing: 1, textTransform: "uppercase", padding: "10px 14px", borderBottom: "1px solid #222", textAlign: "left" },
    td: { padding: "9px 14px", borderBottom: "1px solid #1a1a1a", color: "#ccc" },
  };

  if (racesLoading) {
    return (
      <div style={{ ...styles.app, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <div style={{ textAlign: "center" }}>
          <div style={{ color: "#e10600", fontSize: 32, marginBottom: 12 }}>🏎️</div>
          <div style={{ color: "#666" }}>Loading race calendar...</div>
        </div>
      </div>
    );
  }

  const gpList = availableRaces[selectedYear] || [];

  return (
    <div style={styles.app}>
      <div style={styles.header}>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={styles.redBar} />
          <div>
            <div style={{ fontSize: 20, fontWeight: 700, letterSpacing: -0.5 }}>F1 Analytics Dashboard</div>
            <div style={{ color: "#666", fontSize: 13, marginTop: 2 }}>{selectedGp} · {selectedYear}</div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <div style={{ color: "#555", fontSize: 11, letterSpacing: 1 }}>SELECT RACE</div>
          <select value={selectedYear} onChange={(e) => { const yr = e.target.value; setSelectedYear(yr); const first = availableRaces[yr]?.[0]; if (first) setSelectedGp(first); }} style={SELECT_STYLE}>
            {Object.keys(availableRaces).sort().map((yr) => <option key={yr} value={yr}>{yr}</option>)}
          </select>
          <select value={selectedGp} onChange={(e) => setSelectedGp(e.target.value)} style={{ ...SELECT_STYLE, minWidth: 240 }}>
            {gpList.map((gp) => <option key={gp} value={gp}>{gp}</option>)}
          </select>
          {dataLoading && (
            <div style={{ color: "#e10600", fontSize: 12, display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#e10600", display: "inline-block" }} />
              Loading race data — first load may take 60s
            </div>
          )}
        </div>
      </div>

      {error && (
        <div style={{ maxWidth: 1200, margin: "24px auto", padding: "0 32px" }}>
          <div style={{ background: "#ff000022", border: "1px solid #ff4444", borderRadius: 8, padding: "12px 16px", color: "#ff6666", fontSize: 13 }}>Error: {error}</div>
        </div>
      )}

      <div style={{ opacity: dataLoading ? 0.4 : 1, transition: "opacity 0.3s", pointerEvents: dataLoading ? "none" : "auto" }}>
        <div style={styles.section}>

          {/* Race overview */}
          <div style={styles.sectionTitle}>Race Overview</div>
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 40, alignItems: "flex-end" }}>
            <FastestLapCard fastestLap={raceOverview?.fastestLap} />
            <Podium podium={raceOverview?.podium} />
          </div>

          {/* Driver selector */}
          <div style={styles.sectionTitle}>Lap Time Comparison</div>
          <p style={{ color: "#666", fontSize: 13, marginBottom: 16, marginTop: -8 }}>Select drivers to compare — same selection applies to all charts below</p>
          <div style={styles.driverGrid}>
            {allDrivers.map((driver, i) => {
              const color = DRIVER_COLORS[i % DRIVER_COLORS.length];
              const active = selectedDrivers.includes(driver);
              const dnf = driverStatus[driver] && !driverStatus[driver].finished;
              return (
                <button key={driver} style={styles.driverBtn(active, color)} onClick={() => toggleDriver(driver)}>
                  {driver}{dnf && <DnfBadge />}
                </button>
              );
            })}
          </div>

          {/* Lap time chart */}
          <div style={styles.chartBox}>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                <XAxis dataKey="LapNumber" stroke="#555" tick={{ fill: "#888", fontSize: 12 }} label={{ value: "Lap Number", position: "insideBottom", offset: -10, fill: "#666", fontSize: 12 }} />
                <YAxis stroke="#555" tick={{ fill: "#888", fontSize: 12 }} domain={["auto", "auto"]} tickFormatter={formatLapTime} label={{ value: "Lap Time", angle: -90, position: "insideLeft", fill: "#666", fontSize: 12 }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ color: "#888", fontSize: 12, paddingTop: 12 }} />
                {selectedDrivers.map((driver) => (
                  <Line key={driver} type="monotone" dataKey={driver} stroke={DRIVER_COLORS[allDrivers.indexOf(driver) % DRIVER_COLORS.length]} dot={false} strokeWidth={2} connectNulls={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Race position chart */}
          <div style={styles.sectionTitle}>Race Position</div>
          <p style={{ color: "#666", fontSize: 13, marginBottom: 16, marginTop: -8 }}>Position each lap — dips show pit stops, climbs show overtakes</p>
          <div style={styles.chartBox}>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={positionChartData} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                <XAxis dataKey="LapNumber" stroke="#555" tick={{ fill: "#888", fontSize: 12 }} label={{ value: "Lap Number", position: "insideBottom", offset: -10, fill: "#666", fontSize: 12 }} />
                <YAxis stroke="#555" tick={{ fill: "#888", fontSize: 12 }} reversed={true} domain={[1, 20]} ticks={[1, 5, 10, 15, 20]} label={{ value: "Position", angle: -90, position: "insideLeft", fill: "#666", fontSize: 12 }} />
                <Tooltip content={({ active, payload, label }) => {
                  if (!active || !payload?.length) return null;
                  return (
                    <div style={{ background: "#1a1a1a", border: "1px solid #444", borderRadius: 6, padding: "10px 14px", fontSize: 13 }}>
                      <div style={{ color: "#aaa", marginBottom: 6 }}>Lap {label}</div>
                      {[...payload].sort((a, b) => a.value - b.value).map((p) => (
                        <div key={p.name} style={{ color: p.color, marginBottom: 2 }}>P{p.value} — {p.name}</div>
                      ))}
                    </div>
                  );
                }} />
                <Legend wrapperStyle={{ color: "#888", fontSize: 12, paddingTop: 12 }} />
                {selectedDrivers.map((driver) => (
                  <Line key={driver} type="monotone" dataKey={driver} stroke={DRIVER_COLORS[allDrivers.indexOf(driver) % DRIVER_COLORS.length]} dot={false} strokeWidth={2} connectNulls={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Tire strategy */}
          <div style={styles.sectionTitle}>Tire Strategy</div>
          <p style={{ color: "#666", fontSize: 13, marginBottom: 16, marginTop: -8 }}>Each bar shows a tire stint — hover for compound and lap range</p>
          <div style={styles.chartBox}>
            <StintChart stints={stints} allDrivers={allDrivers} driverStatus={driverStatus} />
          </div>

          {/* AI Lap Time Predictor */}
          <div style={styles.sectionTitle}>AI Lap Time Predictor</div>
          <p style={{ color: "#666", fontSize: 13, marginBottom: 20, marginTop: -8 }}>
            XGBoost model trained on 11,000+ laps — predict lap time for any compound, tyre age, and lap number
          </p>
          <div style={{ marginBottom: 40 }}>
            <LapTimePredictor laps={laps} modelInfo={modelInfo} />
          </div>

          {/* Strategy Simulator */}
          <div style={styles.sectionTitle}>Pit Stop Strategy Simulator</div>
          <p style={{ color: "#666", fontSize: 13, marginBottom: 20, marginTop: -8 }}>
            Compare a driver's actual strategy against a custom one — powered by the ML model
          </p>
          <div style={{ marginBottom: 40 }}>
            <StrategySimulator
              allDrivers={allDrivers}
              driverStatus={driverStatus}
              selectedYear={selectedYear}
              selectedGp={selectedGp}
            />
          </div>

          {/* Driver lap detail */}
          <div style={styles.sectionTitle}>Driver Lap Detail</div>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 20 }}>
            {fastestLap && (
              <div style={{ background: "#1a1a1a", border: "1px solid #333", borderRadius: 8, padding: "14px 20px", minWidth: 140 }}>
                <div style={{ color: "#888", fontSize: 11, textTransform: "uppercase", letterSpacing: 1 }}>Fastest Lap · {tableDriver}</div>
                <div style={{ color: "#fff", fontSize: 22, fontWeight: 700, marginTop: 4, fontFamily: "monospace" }}>{formatLapTime(fastestLap)}</div>
              </div>
            )}
            {avgLap && (
              <div style={{ background: "#1a1a1a", border: "1px solid #333", borderRadius: 8, padding: "14px 20px", minWidth: 140 }}>
                <div style={{ color: "#888", fontSize: 11, textTransform: "uppercase", letterSpacing: 1 }}>Avg Lap · {tableDriver}</div>
                <div style={{ color: "#fff", fontSize: 22, fontWeight: 700, marginTop: 4, fontFamily: "monospace" }}>{formatLapTime(avgLap)}</div>
              </div>
            )}
            {driverStatus[tableDriver] && !driverStatus[tableDriver].finished && (
              <div style={{ background: "#ff000022", border: "1px solid #ff4444", borderRadius: 8, padding: "14px 20px", color: "#ff6666", fontSize: 13, fontWeight: 700, display: "flex", alignItems: "center" }}>
                DNF — {driverStatus[tableDriver].status}
              </div>
            )}
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 20 }}>
            {allDrivers.map((driver, i) => {
              const color = DRIVER_COLORS[i % DRIVER_COLORS.length];
              const dnf = driverStatus[driver] && !driverStatus[driver].finished;
              return (
                <button key={driver} style={styles.driverBtn(tableDriver === driver, color)} onClick={() => setTableDriver(driver)}>
                  {driver}{dnf && <DnfBadge />}
                </button>
              );
            })}
          </div>

          <div style={{ background: "#141414", border: "1px solid #222", borderRadius: 10, overflow: "hidden" }}>
            {tableLaps.length === 0 ? (
              <div style={{ padding: "32px", textAlign: "center", color: "#555" }}>
                No lap data for {tableDriver}
                {driverStatus[tableDriver] && !driverStatus[tableDriver].finished && (
                  <span style={{ color: "#ff6666", marginLeft: 8 }}>({driverStatus[tableDriver].status})</span>
                )}
              </div>
            ) : (
              <table style={styles.table}>
                <thead>
                  <tr>
                    <th style={styles.th}>Lap</th>
                    <th style={styles.th}>Lap Time</th>
                    <th style={styles.th}>Compound</th>
                    <th style={styles.th}>Tyre Life</th>
                    <th style={styles.th}>Δ Fastest</th>
                  </tr>
                </thead>
                <tbody>
                  {tableLaps.map((lap, i) => {
                    const delta = fastestLap ? (lap.LapTimeSeconds - fastestLap).toFixed(3) : null;
                    const isFastest = lap.LapTimeSeconds === fastestLap;
                    return (
                      <tr key={i} style={{ background: isFastest ? "#e1060012" : "transparent" }}>
                        <td style={styles.td}>{lap.LapNumber}</td>
                        <td style={{ ...styles.td, color: isFastest ? "#e10600" : "#ccc", fontWeight: isFastest ? 700 : 400, fontFamily: "monospace" }}>
                          {formatLapTime(lap.LapTimeSeconds)}
                          {isFastest && <span style={{ marginLeft: 8, fontSize: 10, color: "#e10600" }}>▼ FASTEST</span>}
                        </td>
                        <td style={styles.td}><CompoundBadge compound={lap.Compound} /></td>
                        <td style={{ ...styles.td, color: "#888" }}>{lap.TyreLife} laps</td>
                        <td style={{ ...styles.td, color: isFastest ? "#e10600" : "#555", fontFamily: "monospace", fontSize: 12 }}>
                          {isFastest ? "—" : `+${delta}s`}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>

        </div>
      </div>
    </div>
  );
}