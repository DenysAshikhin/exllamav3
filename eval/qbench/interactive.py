"""
Interactive HTML version of the combined per-token KLD histogram.

Same curves as the static chart -- both consume plot.combined_hist_series, so the binning,
p1-p99 trim and drape cannot drift apart -- with the models toggleable. Filtering rescales the
y axis to whatever is still visible, which is the point: dropping a dominant curve lets the
remaining ones use the full height.

Self-contained: no external scripts, styles or fonts, so it opens straight off disk.
"""
import json

from .plot import combined_hist_series


def _hex(rgb):
    r, g, b = (int(round(255 * float(c))) for c in rgb[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def write_kld_hist_html(
    entries: list, title: str, subtitle: str, dark: bool, html_file: str,
    caption: str | bool = True, x_log: bool = True, y_log: bool = False,
    ref_desc: str = "bf16",
) -> bool:
    """Writes the interactive histogram. Returns False when there is nothing plottable."""
    computed = combined_hist_series(entries, x_log = x_log, y_log = y_log)
    if computed is None:
        return False

    centers = [float(x) for x in computed["centers"]]
    series = [
        {
            "label": s["label"],
            "group": s["group"],
            "color": _hex(s["color"]),
            "a": s["a"],
            "y": [float(v) for v in s["y"]],
        }
        for s in computed["series"]
    ]
    floor = computed["floor"]
    payload = {
        "title": title,
        "subtitle": subtitle or "",
        "caption": (
            "Per-token KL divergence against the unquantized reference, histogram per model: "
            f"{len(centers)} shared bins, uniform on the {'log ' if x_log else ''}x axis, cropped "
            "to the models' p1-p99 range. The dotted line is the noise floor: the divergence of "
            f"the {ref_desc} reference against itself under rounding-scale perturbation; its "
            "lower tail may run off the left edge. Toggle models to rescale the y axis to what "
            "remains."
        ) if caption is not False else "",
        "centers": centers,
        "series": series,
        "floor": ({"a": floor["a"], "y": [float(v) for v in floor["y"]]} if floor else None),
        "xLog": bool(x_log),
        "yLog": bool(y_log),
        "dark": bool(dark),
    }

    html = _TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))
    with open(html_file, "w", encoding = "utf-8") as f:
        f.write(html)
    return True


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>qbench - per-token KLD histogram</title>
<style>
  :root {
    --bg: #1b1d23; --panel: #23262e; --fg: #e8e9ec; --muted: #9aa0ac;
    --grid: #343842; --floor: #9aa0ac; --border: #333741;
  }
  html.light {
    --bg: #ffffff; --panel: #f5f6f8; --fg: #1b1d23; --muted: #5b616e;
    --grid: #e3e5ea; --floor: #6b7280; --border: #d8dbe1;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  .wrap { max-width: 1500px; margin: 0 auto; padding: 20px; }
  header { text-align: center; margin-bottom: 8px; }
  h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; }
  .sub { color: var(--muted); font-size: 14px; }
  .layout { display: flex; gap: 18px; align-items: flex-start; flex-wrap: wrap; }
  .chart { flex: 1 1 720px; min-width: 320px; }
  svg { width: 100%; height: auto; display: block; overflow: visible; }
  .panel {
    flex: 0 0 210px; background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px;
  }
  .panel h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .06em;
    color: var(--muted); margin: 0 0 10px; font-weight: 600; }
  .row { display: flex; align-items: center; gap: 8px; padding: 3px 0; cursor: pointer;
    user-select: none; }
  .row input { margin: 0; cursor: pointer; flex: none; }
  .sw { width: 22px; height: 3px; border-radius: 2px; flex: none; }
  .row.off .name { opacity: .45; }
  .name { font-variant-numeric: tabular-nums; }
  .btns { display: flex; gap: 6px; margin: 10px 0 4px; }
  button {
    flex: 1; background: transparent; color: var(--fg); border: 1px solid var(--border);
    border-radius: 6px; padding: 5px 8px; font-size: 12px; cursor: pointer; font-family: inherit;
  }
  button:hover { border-color: var(--muted); }
  .cap { color: var(--muted); font-size: 12px; margin-top: 14px; max-width: 1100px; }
  .axis { fill: var(--muted); font-size: 12px; }
  .axis-title { fill: var(--fg); font-size: 13px; }
  .gridline { stroke: var(--grid); stroke-width: 1; }
  #tip {
    position: fixed; pointer-events: none; background: var(--panel); color: var(--fg);
    border: 1px solid var(--border); border-radius: 6px; padding: 7px 9px; font-size: 12px;
    opacity: 0; transition: opacity .1s; z-index: 10; font-variant-numeric: tabular-nums;
    max-height: 60vh; overflow: hidden; box-shadow: 0 4px 14px rgba(0,0,0,.28);
  }
  #tip .t { display: flex; align-items: center; gap: 6px; }
  #tip b { font-weight: 600; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1 id="title"></h1>
    <div class="sub" id="subtitle"></div>
  </header>
  <div class="layout">
    <div class="chart">
      <svg id="chart" viewBox="0 0 1000 620" preserveAspectRatio="xMidYMid meet"
           role="img" aria-label="Per-token KL divergence histogram"></svg>
    </div>
    <aside class="panel">
      <h2>Models</h2>
      <div class="btns">
        <button type="button" id="all">All</button>
        <button type="button" id="none">None</button>
        <button type="button" id="theme">Theme</button>
      </div>
      <div id="keys"></div>
    </aside>
  </div>
  <div class="cap" id="caption"></div>
</div>
<div id="tip"></div>
<script>
const DATA = __PAYLOAD__;

if (!DATA.dark) document.documentElement.classList.add("light");
document.getElementById("title").textContent = DATA.title;
document.getElementById("subtitle").textContent = DATA.subtitle;
document.getElementById("caption").textContent = DATA.caption;

const W = 1000, H = 620;
const M = { top: 16, right: 18, bottom: 62, left: 74 };
const IW = W - M.left - M.right, IH = H - M.top - M.bottom;
const svg = document.getElementById("chart");
const tip = document.getElementById("tip");
const NS = "http://www.w3.org/2000/svg";

const visible = new Set(DATA.series.map(s => s.label));
let floorOn = DATA.floor !== null;

const xs = DATA.centers;
const fwd = DATA.xLog ? (v => Math.log10(Math.max(v, 1e-12))) : (v => v);
const x0 = fwd(xs[0]), x1 = fwd(xs[xs.length - 1]);
const xPix = v => M.left + (fwd(v) - x0) / (x1 - x0) * IW;

function el(name, attrs, text) {
  const n = document.createElementNS(NS, name);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (text !== undefined) n.textContent = text;
  return n;
}

function niceTicks(lo, hi, count) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(v);
  return out;
}

// The static chart takes its y limit from the models only; the floor may clip. Same here, and
// it is recomputed from the VISIBLE models so filtering reclaims the height.
function yTop() {
  let peak = 0;
  for (const s of DATA.series) {
    if (!visible.has(s.label)) continue;
    for (const v of s.y) if (v > peak) peak = v;
  }
  if (peak <= 0) peak = 1;
  return DATA.yLog ? peak * 2.6 : peak * 1.22;
}

function render() {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const top = yTop();
  const yBase = DATA.yLog ? Math.log10(0.8) : 0;
  const yMax = DATA.yLog ? Math.log10(top) : top;
  const yPix = v => {
    const t = DATA.yLog ? Math.log10(Math.max(v, 0.8)) : v;
    return M.top + IH - (t - yBase) / (yMax - yBase) * IH;
  };

  // grid + x ticks (decades on a log axis, nice steps otherwise)
  const xTicks = [];
  if (DATA.xLog) {
    for (let k = Math.ceil(x0); k <= Math.floor(x1); k++) xTicks.push({ v: Math.pow(10, k), k });
  } else {
    for (const v of niceTicks(x0, x1, 8)) xTicks.push({ v, k: null });
  }
  for (const t of xTicks) {
    const px = xPix(t.v);
    svg.appendChild(el("line", { x1: px, y1: M.top, x2: px, y2: M.top + IH, class: "gridline" }));
    const g = el("text", { x: px, y: M.top + IH + 20, "text-anchor": "middle", class: "axis" });
    if (t.k === null) g.textContent = String(Number(t.v.toPrecision(3)));
    else {
      g.appendChild(el("tspan", {}, "10"));
      const sup = el("tspan", { dy: "-6", "font-size": "9" }, String(t.k));
      g.appendChild(sup);
    }
    svg.appendChild(g);
  }
  for (const v of niceTicks(DATA.yLog ? 0 : 0, top, 6)) {
    if (v < 0) continue;
    const py = yPix(v);
    if (py < M.top - 1 || py > M.top + IH + 1) continue;
    svg.appendChild(el("line", { x1: M.left, y1: py, x2: M.left + IW, y2: py, class: "gridline" }));
    svg.appendChild(el("text", {
      x: M.left - 10, y: py + 4, "text-anchor": "end", class: "axis",
    }, String(Math.round(v))));
  }

  svg.appendChild(el("text", {
    x: M.left + IW / 2, y: H - 16, "text-anchor": "middle", class: "axis-title",
  }, "per-token KL divergence" + (DATA.xLog ? " (log)" : "")));
  svg.appendChild(el("text", {
    x: 18, y: M.top + IH / 2, "text-anchor": "middle", class: "axis-title",
    transform: `rotate(-90 18 ${M.top + IH / 2})`,
  }, "tokens per bin"));

  const path = (a, ys) => ys.map((v, i) =>
    `${i ? "L" : "M"}${xPix(xs[a + i]).toFixed(2)},${yPix(v).toFixed(2)}`).join("");

  if (floorOn && DATA.floor) {
    svg.appendChild(el("path", {
      d: path(DATA.floor.a, DATA.floor.y), fill: "none",
      stroke: getComputedStyle(document.documentElement).getPropertyValue("--floor").trim(),
      "stroke-width": 1.8, "stroke-dasharray": "3 3", "stroke-linejoin": "round",
    }));
  }
  for (const s of DATA.series) {
    if (!visible.has(s.label)) continue;
    svg.appendChild(el("path", {
      d: path(s.a, s.y), fill: "none", stroke: s.color, "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round",
    }));
  }
  svg.appendChild(el("rect", {
    x: M.left, y: M.top, width: IW, height: IH, fill: "transparent", id: "hit",
  }));
}

function buildKeys() {
  const box = document.getElementById("keys");
  box.textContent = "";
  const add = (label, color, dashed, checked, onChange) => {
    const row = document.createElement("label");
    row.className = "row" + (checked ? "" : " off");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = checked;
    cb.addEventListener("change", () => {
      row.classList.toggle("off", !cb.checked);
      onChange(cb.checked);
      render();
    });
    const sw = document.createElement("span");
    sw.className = "sw";
    sw.style.background = dashed
      ? `repeating-linear-gradient(90deg, ${color} 0 4px, transparent 4px 7px)`
      : color;
    const nm = document.createElement("span");
    nm.className = "name";
    nm.textContent = label;
    row.append(cb, sw, nm);
    box.appendChild(row);
    return cb;
  };
  const boxes = [];
  for (const s of DATA.series) {
    boxes.push({
      cb: add(s.label, s.color, false, visible.has(s.label), on => {
        if (on) visible.add(s.label); else visible.delete(s.label);
      }),
      label: s.label,
    });
  }
  if (DATA.floor) {
    const c = getComputedStyle(document.documentElement).getPropertyValue("--floor").trim();
    add("noise floor", c || "#9aa0ac", true, floorOn, on => { floorOn = on; });
  }
  return boxes;
}

let boxes = buildKeys();
render();

document.getElementById("all").addEventListener("click", () => {
  for (const s of DATA.series) visible.add(s.label);
  floorOn = DATA.floor !== null;
  boxes = buildKeys();
  render();
});
document.getElementById("none").addEventListener("click", () => {
  visible.clear();
  floorOn = false;
  boxes = buildKeys();
  render();
});
document.getElementById("theme").addEventListener("click", () => {
  document.documentElement.classList.toggle("light");
  boxes = buildKeys();
  render();
});

svg.addEventListener("mousemove", ev => {
  const r = svg.getBoundingClientRect();
  const px = (ev.clientX - r.left) / r.width * W;
  if (px < M.left || px > M.left + IW) { tip.style.opacity = 0; return; }
  const t = x0 + (px - M.left) / IW * (x1 - x0);
  const target = DATA.xLog ? Math.pow(10, t) : t;
  let bin = 0, best = Infinity;
  for (let i = 0; i < xs.length; i++) {
    const d = Math.abs(fwd(xs[i]) - fwd(target));
    if (d < best) { best = d; bin = i; }
  }
  const shown = DATA.series
    .filter(s => visible.has(s.label) && bin >= s.a && bin < s.a + s.y.length)
    .map(s => ({ label: s.label, color: s.color, v: s.y[bin - s.a] }))
    .sort((p, q) => q.v - p.v)
    .slice(0, 14);
  if (!shown.length) { tip.style.opacity = 0; return; }
  const head = `<b>KLD &asymp; ${Number(xs[bin].toPrecision(3))}</b>`;
  tip.innerHTML = head + shown.map(s =>
    `<div class="t"><span class="sw" style="background:${s.color}"></span>` +
    `${s.label}: ${Math.round(s.v)}</div>`).join("");
  tip.style.opacity = 1;
  const pad = 14;
  let left = ev.clientX + pad, tw = tip.offsetWidth;
  if (left + tw > window.innerWidth - 8) left = ev.clientX - tw - pad;
  let topPx = ev.clientY + pad, th = tip.offsetHeight;
  if (topPx + th > window.innerHeight - 8) topPx = Math.max(8, ev.clientY - th - pad);
  tip.style.left = left + "px";
  tip.style.top = topPx + "px";
});
svg.addEventListener("mouseleave", () => { tip.style.opacity = 0; });
</script>
</body>
</html>
"""
