/* ══════════════════════════════════════════════════════════════════
   Bayesian Baseball Projections — D3 Dashboard
   ══════════════════════════════════════════════════════════════════ */

const STAT_LABELS = {
  k_rate: "K%", bb_rate: "BB%", hr_rate: "HR/PA", iso: "ISO", babip: "BABIP",
  avg: "AVG", obp: "OBP", slg: "SLG", woba: "wOBA", wrc_plus: "wRC+", off: "Off"
};

const SYSTEMS = [
  { key: "our", label: "Bayesian", color: "#4f8ff7" },
  { key: "stea", label: "Steamer", color: "#f59e0b" },
  { key: "zips", label: "ZiPS", color: "#34d399" },
  { key: "dept", label: "Depth Charts", color: "#f87171" },
];

const AGING_COLORS = {
  k_rate: "#f87171", bb_rate: "#4f8ff7", hr_rate: "#f59e0b",
  iso: "#a78bfa", babip: "#34d399"
};

let DATA = {};
let tooltip;

// ─── Data Loading ────────────────────────────────────────────────
async function loadData() {
  const [comparison, ourModel, agingCurves, summary] = await Promise.all([
    d3.json("data/comparison.json"),
    d3.json("data/our_model.json"),
    d3.json("data/aging_curves.json"),
    d3.json("data/summary.json"),
  ]);
  DATA = { comparison, ourModel, agingCurves, summary };
}

// ─── Navigation ──────────────────────────────────────────────────
function initNav() {
  document.querySelectorAll("[data-page]").forEach(link => {
    link.addEventListener("click", e => {
      e.preventDefault();
      const page = link.dataset.page;
      document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
      document.querySelectorAll("[data-page]").forEach(l => l.classList.remove("active"));
      document.getElementById(`page-${page}`).classList.add("active");
      link.classList.add("active");
      renderPage(page);
    });
  });
}

const rendered = {};
function renderPage(page) {
  if (rendered[page]) return;
  rendered[page] = true;
  switch (page) {
    case "overview": renderOverview(); break;
    case "player": renderPlayerPage(); break;
    case "comparison": renderComparison(); break;
    case "aging": renderAging(); break;
    case "leaderboard": renderLeaderboard(); break;
  }
}

// ─── Tooltip ─────────────────────────────────────────────────────
function initTooltip() {
  tooltip = d3.select("body").append("div").attr("class", "tooltip").style("display", "none");
}

function showTooltip(evt, html) {
  tooltip.html(html).style("display", "block")
    .style("left", (evt.clientX + 12) + "px")
    .style("top", (evt.clientY - 10) + "px");
}

function hideTooltip() {
  tooltip.style("display", "none");
}

// ─── Formatting ──────────────────────────────────────────────────
function fmt(v, stat) {
  if (v === "" || v == null || isNaN(v)) return "—";
  if (stat === "wrc_plus") return Math.round(v);
  if (stat === "off" || stat === "wraa") return (+v).toFixed(1);
  return (+v).toFixed(3);
}

// ══════════════════════════════════════════════════════════════════
// OVERVIEW PAGE
// ══════════════════════════════════════════════════════════════════
function renderOverview() {
  const { comparison, summary } = DATA;

  // Metrics
  const corr_stea = summary.correlations?.woba?.Steamer || 0;
  const corr_zips = summary.correlations?.woba?.ZiPS || 0;
  const metricsHtml = [
    { label: "Players Projected", value: summary.total_players },
    { label: "Matched vs FanGraphs", value: summary.matched_players },
    { label: "wOBA Corr vs Steamer", value: corr_stea.toFixed(3) },
    { label: "wOBA Corr vs ZiPS", value: corr_zips.toFixed(3) },
  ].map(m => `<div class="metric-card"><div class="label">${m.label}</div><div class="value">${m.value}</div></div>`).join("");
  document.getElementById("overview-metrics").innerHTML = metricsHtml;

  // System averages table
  const stats = ["k_rate", "bb_rate", "hr_rate", "iso", "babip", "avg", "obp", "slg", "woba", "wrc_plus", "off"];
  let tableHtml = "<table><thead><tr><th>Stat</th>";
  SYSTEMS.forEach(s => tableHtml += `<th style="color:${s.color}">${s.label}</th>`);
  tableHtml += "</tr></thead><tbody>";
  stats.forEach(stat => {
    tableHtml += `<tr><td class="name-cell">${STAT_LABELS[stat]}</td>`;
    SYSTEMS.forEach(sys => {
      const d = summary.systems?.[sys.label]?.[stat];
      const val = d ? fmt(d.mean, stat) : "—";
      tableHtml += `<td>${val}</td>`;
    });
    tableHtml += "</tr>";
  });
  tableHtml += "</tbody></table>";
  document.getElementById("system-averages-table").innerHTML = tableHtml;

  // wOBA scatter
  const scatterData = comparison.filter(d => d.our_woba && d.stea_woba);
  renderScatter("#woba-scatter", scatterData, "stea_woba", "our_woba", "Steamer wOBA", "Our Model wOBA", 600, 450);

  // Disagreements
  const disagree = scatterData.map(d => ({
    ...d, diff: d.our_woba - d.stea_woba, abs_diff: Math.abs(d.our_woba - d.stea_woba)
  })).sort((a, b) => b.abs_diff - a.abs_diff);

  const bullish = disagree.filter(d => d.diff > 0).slice(0, 10);
  const bearish = disagree.filter(d => d.diff < 0).slice(0, 10);

  document.getElementById("bullish-list").innerHTML = bullish.map(d =>
    `<div class="disagree-item">
      <div><span class="name">${d.name}</span> <span class="meta">${d.team}, ${Math.round(d.age)}</span></div>
      <div class="values">Ours ${fmt(d.our_woba, "woba")} vs ${fmt(d.stea_woba, "woba")} <span class="diff-pos">+${d.diff.toFixed(3)}</span></div>
    </div>`
  ).join("");

  document.getElementById("bearish-list").innerHTML = bearish.map(d =>
    `<div class="disagree-item">
      <div><span class="name">${d.name}</span> <span class="meta">${d.team}, ${Math.round(d.age)}</span></div>
      <div class="values">Ours ${fmt(d.our_woba, "woba")} vs ${fmt(d.stea_woba, "woba")} <span class="diff-neg">${d.diff.toFixed(3)}</span></div>
    </div>`
  ).join("");
}

// ══════════════════════════════════════════════════════════════════
// D3 SCATTER PLOT
// ══════════════════════════════════════════════════════════════════
function renderScatter(selector, data, xKey, yKey, xLabel, yLabel, width = 500, height = 400) {
  const container = d3.select(selector);
  container.selectAll("*").remove();
  const margin = { top: 20, right: 20, bottom: 45, left: 55 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  const svg = container.append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%");
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  const xVals = data.map(d => +d[xKey]).filter(v => !isNaN(v));
  const yVals = data.map(d => +d[yKey]).filter(v => !isNaN(v));
  const pad = 0.005;
  const allVals = xVals.concat(yVals);
  const lo = d3.min(allVals) - pad;
  const hi = d3.max(allVals) + pad;

  const x = d3.scaleLinear().domain([lo, hi]).range([0, w]);
  const y = d3.scaleLinear().domain([lo, hi]).range([h, 0]);

  const ageExtent = d3.extent(data, d => +d.age);
  const colorScale = d3.scaleSequential(d3.interpolateViridis).domain(ageExtent);

  // Axes
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).ticks(8));
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(8));

  // Axis labels
  svg.append("text").attr("x", width / 2).attr("y", height - 5).attr("text-anchor", "middle")
    .attr("fill", "#8b8fa3").attr("font-size", 12).text(xLabel);
  svg.append("text").attr("transform", "rotate(-90)").attr("x", -height / 2).attr("y", 14)
    .attr("text-anchor", "middle").attr("fill", "#8b8fa3").attr("font-size", 12).text(yLabel);

  // y=x line
  g.append("line").attr("x1", x(lo)).attr("y1", y(lo)).attr("x2", x(hi)).attr("y2", y(hi))
    .attr("stroke", "#3a3d4a").attr("stroke-dasharray", "4,4");

  // Points
  g.selectAll("circle").data(data).enter().append("circle")
    .attr("cx", d => x(+d[xKey]))
    .attr("cy", d => y(+d[yKey]))
    .attr("r", 4)
    .attr("fill", d => colorScale(+d.age))
    .attr("opacity", 0.75)
    .attr("stroke", "none")
    .on("mouseenter", function (evt, d) {
      d3.select(this).attr("r", 7).attr("stroke", "#fff").attr("stroke-width", 1.5);
      showTooltip(evt,
        `<div class="tt-name">${d.name}</div>
         <div class="tt-dim">${d.team} · Age ${Math.round(d.age)}</div>
         <div>Ours: ${fmt(d[yKey], "woba")} · Other: ${fmt(d[xKey], "woba")}</div>`
      );
    })
    .on("mouseleave", function () {
      d3.select(this).attr("r", 4).attr("stroke", "none");
      hideTooltip();
    });
}

// ══════════════════════════════════════════════════════════════════
// MINI SCATTER (for comparison page)
// ══════════════════════════════════════════════════════════════════
function renderMiniScatter(selector, data, xKey, yKey, color, stat) {
  const container = d3.select(selector);
  container.selectAll("*").remove();
  const width = 320, height = 280;
  const margin = { top: 10, right: 10, bottom: 35, left: 45 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  const filtered = data.filter(d => d[xKey] !== "" && d[yKey] !== "" && !isNaN(+d[xKey]) && !isNaN(+d[yKey]));
  if (!filtered.length) { container.append("p").text("No data").style("color", "#5c6078"); return; }

  const svg = container.append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%");
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  const allVals = filtered.flatMap(d => [+d[xKey], +d[yKey]]);
  const pad = (d3.max(allVals) - d3.min(allVals)) * 0.05;
  const lo = d3.min(allVals) - pad, hi = d3.max(allVals) + pad;

  const x = d3.scaleLinear().domain([lo, hi]).range([0, w]);
  const y = d3.scaleLinear().domain([lo, hi]).range([h, 0]);

  g.append("g").attr("class", "axis").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).ticks(5));
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5));
  g.append("line").attr("x1", x(lo)).attr("y1", y(lo)).attr("x2", x(hi)).attr("y2", y(hi))
    .attr("stroke", "#3a3d4a").attr("stroke-dasharray", "4,4");

  // Correlation + MAE
  const xArr = filtered.map(d => +d[xKey]), yArr = filtered.map(d => +d[yKey]);
  const corr = pearsonCorr(xArr, yArr);
  const mae = d3.mean(filtered, d => Math.abs(+d[yKey] - +d[xKey]));
  svg.append("text").attr("x", margin.left + 4).attr("y", margin.top + 12)
    .attr("fill", "#8b8fa3").attr("font-size", 11)
    .text(`r=${corr.toFixed(3)}  MAE=${mae.toFixed(3)}`);

  g.selectAll("circle").data(filtered).enter().append("circle")
    .attr("cx", d => x(+d[xKey])).attr("cy", d => y(+d[yKey]))
    .attr("r", 3).attr("fill", color).attr("opacity", 0.5)
    .on("mouseenter", function (evt, d) {
      d3.select(this).attr("r", 6).attr("opacity", 1);
      showTooltip(evt, `<div class="tt-name">${d.name}</div><div>Ours: ${fmt(d[yKey], stat)} · Other: ${fmt(d[xKey], stat)}</div>`);
    })
    .on("mouseleave", function () { d3.select(this).attr("r", 3).attr("opacity", 0.5); hideTooltip(); });
}

function pearsonCorr(x, y) {
  const n = x.length;
  const mx = d3.mean(x), my = d3.mean(y);
  let num = 0, dx2 = 0, dy2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - mx, dy = y[i] - my;
    num += dx * dy; dx2 += dx * dx; dy2 += dy * dy;
  }
  return num / Math.sqrt(dx2 * dy2);
}

// ══════════════════════════════════════════════════════════════════
// PLAYER LOOKUP PAGE
// ══════════════════════════════════════════════════════════════════
function renderPlayerPage() {
  const input = document.getElementById("player-search");
  const sugDiv = document.getElementById("player-suggestions");
  const players = DATA.comparison.filter(d => d.name && d.name !== "");

  input.addEventListener("input", () => {
    const q = input.value.toLowerCase();
    if (q.length < 2) { sugDiv.classList.remove("visible"); return; }
    const matches = players.filter(d => d.name.toLowerCase().includes(q)).slice(0, 12);
    sugDiv.innerHTML = matches.map(d =>
      `<div class="suggestion-item" data-batter="${d.batter}">
        <span>${d.name}</span><span class="team">${d.team} · ${Math.round(d.age)}</span>
      </div>`
    ).join("");
    sugDiv.classList.add("visible");
    sugDiv.querySelectorAll(".suggestion-item").forEach(el => {
      el.addEventListener("click", () => {
        const batter = +el.dataset.batter;
        input.value = el.querySelector("span").textContent;
        sugDiv.classList.remove("visible");
        renderPlayerCard(players.find(d => d.batter === batter));
      });
    });
  });

  input.addEventListener("blur", () => setTimeout(() => sugDiv.classList.remove("visible"), 200));

  // Default: show first player
  if (players.length) {
    const defaultPlayer = players.sort((a, b) => (b.our_woba || 0) - (a.our_woba || 0))[0];
    input.value = defaultPlayer.name;
    renderPlayerCard(defaultPlayer);
  }
}

function renderPlayerCard(player) {
  const container = document.getElementById("player-card");

  // Header
  let html = `
    <div class="player-header">
      <h2>${player.name}</h2>
      <div class="player-meta">
        <span>${player.team}</span>
        <span>Age ${Math.round(player.age)}</span>
        <span>Bats ${player.stand}</span>
      </div>
    </div>
    <div class="grid-2">
      <div class="card"><h3>Component Rates</h3><div id="player-components"></div></div>
      <div class="card"><h3>Projection Uncertainty</h3><div id="player-uncertainty"></div></div>
    </div>
    <div class="card"><h3>Aggregate Projections</h3><div id="player-agg-table"></div></div>
  `;
  container.innerHTML = html;

  // Component bar chart
  renderComponentBars(player);
  renderUncertainty(player);
  renderPlayerAggTable(player);
}

function renderComponentBars(player) {
  const stats = ["k_rate", "bb_rate", "hr_rate", "iso", "babip"];
  const width = 450, height = 280;
  const margin = { top: 10, right: 20, bottom: 30, left: 55 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  const container = d3.select("#player-components");
  container.selectAll("*").remove();
  const svg = container.append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%");
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  const groups = stats.map(stat => STAT_LABELS[stat]);
  const x0 = d3.scaleBand().domain(groups).range([0, w]).padding(0.25);
  const x1 = d3.scaleBand().domain(SYSTEMS.map(s => s.key)).range([0, x0.bandwidth()]).padding(0.05);

  const allVals = [];
  stats.forEach(stat => {
    SYSTEMS.forEach(sys => {
      const v = +player[`${sys.key}_${stat}`];
      if (!isNaN(v)) allVals.push(v);
    });
  });

  const y = d3.scaleLinear().domain([0, d3.max(allVals) * 1.1]).range([h, 0]);

  g.append("g").attr("class", "axis").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x0));
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(6));

  stats.forEach((stat, i) => {
    const group = g.append("g").attr("transform", `translate(${x0(STAT_LABELS[stat])},0)`);
    SYSTEMS.forEach(sys => {
      const val = +player[`${sys.key}_${stat}`];
      if (isNaN(val)) return;
      group.append("rect")
        .attr("x", x1(sys.key))
        .attr("y", y(val))
        .attr("width", x1.bandwidth())
        .attr("height", h - y(val))
        .attr("fill", sys.color)
        .attr("rx", 2)
        .attr("opacity", 0.85);
    });
  });

  // Legend
  const legend = svg.append("g").attr("transform", `translate(${margin.left + 10}, ${margin.top})`);
  SYSTEMS.forEach((sys, i) => {
    legend.append("rect").attr("x", i * 90).attr("y", 0).attr("width", 10).attr("height", 10).attr("fill", sys.color).attr("rx", 2);
    legend.append("text").attr("x", i * 90 + 14).attr("y", 9).attr("fill", "#8b8fa3").attr("font-size", 10).text(sys.label);
  });
}

function renderUncertainty(player) {
  const stats = ["k_rate", "bb_rate", "hr_rate", "iso", "babip"];
  const width = 450, height = 280;
  const margin = { top: 10, right: 20, bottom: 30, left: 55 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  const container = d3.select("#player-uncertainty");
  container.selectAll("*").remove();
  const svg = container.append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%");
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  const items = stats.map(stat => {
    const val = +player[`our_${stat}`];
    const std = +player[`our_${stat}_std`];
    return { stat, label: STAT_LABELS[stat], val, std: isNaN(std) ? 0 : std };
  }).filter(d => !isNaN(d.val));

  const x = d3.scaleBand().domain(items.map(d => d.label)).range([0, w]).padding(0.4);
  const allVals = items.flatMap(d => [d.val - 2 * d.std, d.val + 2 * d.std]);
  const y = d3.scaleLinear().domain([d3.min(allVals) * 0.95, d3.max(allVals) * 1.05]).range([h, 0]);

  g.append("g").attr("class", "axis").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x));
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(6));

  items.forEach(d => {
    const cx = x(d.label) + x.bandwidth() / 2;
    // 2σ band
    g.append("rect")
      .attr("x", cx - 12).attr("width", 24)
      .attr("y", y(d.val + 2 * d.std)).attr("height", y(d.val - 2 * d.std) - y(d.val + 2 * d.std))
      .attr("fill", "#4f8ff7").attr("opacity", 0.1).attr("rx", 4);
    // 1σ band
    g.append("rect")
      .attr("x", cx - 12).attr("width", 24)
      .attr("y", y(d.val + d.std)).attr("height", y(d.val - d.std) - y(d.val + d.std))
      .attr("fill", "#4f8ff7").attr("opacity", 0.25).attr("rx", 4);
    // Point
    g.append("circle").attr("cx", cx).attr("cy", y(d.val)).attr("r", 5).attr("fill", "#4f8ff7");
  });
}

function renderPlayerAggTable(player) {
  const stats = ["avg", "obp", "slg", "woba", "wrc_plus", "off"];
  let html = "<table><thead><tr><th>System</th>";
  stats.forEach(s => html += `<th>${STAT_LABELS[s]}</th>`);
  html += "<th>HR</th></tr></thead><tbody>";

  SYSTEMS.forEach(sys => {
    html += `<tr><td class="name-cell" style="color:${sys.color}">${sys.label}</td>`;
    stats.forEach(stat => {
      const val = player[`${sys.key}_${stat}`];
      html += `<td>${fmt(val, stat)}</td>`;
    });
    const hr = player[`${sys.key}_hr`];
    html += `<td>${hr !== "" && hr != null ? Math.round(hr) : "—"}</td></tr>`;
  });

  html += "</tbody></table>";
  document.getElementById("player-agg-table").innerHTML = html;
}

// ══════════════════════════════════════════════════════════════════
// SYSTEM COMPARISON PAGE
// ══════════════════════════════════════════════════════════════════
function renderComparison() {
  const select = document.getElementById("comparison-stat");
  const render = () => renderComparisonStat(select.value);
  select.addEventListener("change", render);
  render();
}

function renderComparisonStat(stat) {
  const { comparison } = DATA;

  // Correlation matrix
  renderCorrMatrix(stat);

  // Distribution
  renderDistribution(stat);

  // Scatter plots
  SYSTEMS.filter(s => s.key !== "our").forEach(sys => {
    const selectorMap = { stea: "#scatter-steamer", zips: "#scatter-zips", dept: "#scatter-dc" };
    renderMiniScatter(selectorMap[sys.key], comparison, `${sys.key}_${stat}`, `our_${stat}`, sys.color, stat);
  });

  // Residuals vs age
  renderResiduals(stat);
}

function renderCorrMatrix(stat) {
  const { comparison } = DATA;
  const container = d3.select("#corr-matrix");
  container.selectAll("*").remove();

  const labels = SYSTEMS.map(s => s.label);
  const n = labels.length;
  const matrix = [];

  for (let i = 0; i < n; i++) {
    matrix[i] = [];
    for (let j = 0; j < n; j++) {
      const ki = `${SYSTEMS[i].key}_${stat}`, kj = `${SYSTEMS[j].key}_${stat}`;
      const pairs = comparison.filter(d => d[ki] !== "" && d[kj] !== "" && !isNaN(+d[ki]) && !isNaN(+d[kj]));
      if (pairs.length > 10) {
        matrix[i][j] = pearsonCorr(pairs.map(d => +d[ki]), pairs.map(d => +d[kj]));
      } else {
        matrix[i][j] = null;
      }
    }
  }

  const size = 60, pad = 80;
  const width = pad + n * size, height = pad + n * size;
  const svg = container.append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("width", Math.min(width, 400));
  const g = svg.append("g").attr("transform", `translate(${pad},${pad})`);

  const color = d3.scaleSequential(d3.interpolateRdBu).domain([0.5, 1]);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (matrix[i][j] == null) continue;
      g.append("rect").attr("x", j * size).attr("y", i * size).attr("width", size - 2).attr("height", size - 2)
        .attr("fill", color(matrix[i][j])).attr("rx", 4);
      g.append("text").attr("x", j * size + size / 2 - 1).attr("y", i * size + size / 2 + 1)
        .attr("text-anchor", "middle").attr("dominant-baseline", "middle")
        .attr("fill", matrix[i][j] > 0.8 ? "#fff" : "#ccc").attr("font-size", 12).attr("font-weight", 600)
        .text(matrix[i][j].toFixed(3));
    }
  }

  // Labels
  labels.forEach((l, i) => {
    svg.append("text").attr("x", pad + i * size + size / 2 - 1).attr("y", pad - 8)
      .attr("text-anchor", "middle").attr("fill", SYSTEMS[i].color).attr("font-size", 11).text(l);
    svg.append("text").attr("x", pad - 8).attr("y", pad + i * size + size / 2 + 1)
      .attr("text-anchor", "end").attr("dominant-baseline", "middle")
      .attr("fill", SYSTEMS[i].color).attr("font-size", 11).text(l);
  });
}

function renderDistribution(stat) {
  const { comparison } = DATA;
  const container = d3.select("#dist-chart");
  container.selectAll("*").remove();
  const width = 700, height = 300;
  const margin = { top: 20, right: 20, bottom: 35, left: 45 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  const svg = container.append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%");
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  const allVals = [];
  SYSTEMS.forEach(sys => {
    comparison.forEach(d => {
      const v = +d[`${sys.key}_${stat}`];
      if (!isNaN(v) && d[`${sys.key}_${stat}`] !== "") allVals.push(v);
    });
  });

  const x = d3.scaleLinear().domain(d3.extent(allVals)).nice().range([0, w]);
  const bins = d3.bin().domain(x.domain()).thresholds(40);

  let maxCount = 0;
  SYSTEMS.forEach(sys => {
    const vals = comparison.map(d => +d[`${sys.key}_${stat}`]).filter(v => !isNaN(v));
    const b = bins(vals);
    b.forEach(bin => { if (bin.length > maxCount) maxCount = bin.length; });
  });

  const y = d3.scaleLinear().domain([0, maxCount]).range([h, 0]);
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).ticks(8));
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5));

  SYSTEMS.forEach(sys => {
    const vals = comparison.map(d => +d[`${sys.key}_${stat}`]).filter(v => !isNaN(v));
    if (!vals.length) return;
    const b = bins(vals);

    // Area/line
    const area = d3.area()
      .x(d => x((d.x0 + d.x1) / 2))
      .y0(h)
      .y1(d => y(d.length))
      .curve(d3.curveBasis);

    g.append("path").datum(b).attr("d", area)
      .attr("fill", sys.color).attr("opacity", 0.15);

    const line = d3.line()
      .x(d => x((d.x0 + d.x1) / 2))
      .y(d => y(d.length))
      .curve(d3.curveBasis);

    g.append("path").datum(b).attr("d", line)
      .attr("stroke", sys.color).attr("stroke-width", 2).attr("fill", "none");
  });

  // Legend
  const legend = svg.append("g").attr("transform", `translate(${width - 180}, ${margin.top + 5})`);
  SYSTEMS.forEach((sys, i) => {
    legend.append("rect").attr("x", 0).attr("y", i * 18).attr("width", 12).attr("height", 3).attr("fill", sys.color);
    legend.append("text").attr("x", 16).attr("y", i * 18 + 4).attr("fill", "#8b8fa3").attr("font-size", 10).text(sys.label);
  });
}

function renderResiduals(stat) {
  const { comparison } = DATA;
  const container = d3.select("#residual-chart");
  container.selectAll("*").remove();

  const data = comparison.filter(d => d[`our_${stat}`] !== "" && d[`stea_${stat}`] !== "")
    .map(d => ({ ...d, residual: +d[`our_${stat}`] - +d[`stea_${stat}`] }))
    .filter(d => !isNaN(d.residual));

  if (!data.length) return;

  const width = 700, height = 350;
  const margin = { top: 20, right: 20, bottom: 40, left: 55 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  const svg = container.append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%");
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  const x = d3.scaleLinear().domain(d3.extent(data, d => +d.age)).range([0, w]);
  const yExt = d3.max(data, d => Math.abs(d.residual));
  const y = d3.scaleLinear().domain([-yExt * 1.1, yExt * 1.1]).range([h, 0]);

  const color = d3.scaleSequential(d3.interpolateRdBu).domain([-yExt, yExt]);

  g.append("g").attr("class", "axis").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).ticks(8));
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(6));
  g.append("line").attr("x1", 0).attr("y1", y(0)).attr("x2", w).attr("y2", y(0))
    .attr("stroke", "#3a3d4a").attr("stroke-dasharray", "4,4");

  svg.append("text").attr("x", width / 2).attr("y", height - 4).attr("text-anchor", "middle")
    .attr("fill", "#8b8fa3").attr("font-size", 12).text("Player Age");
  svg.append("text").attr("transform", "rotate(-90)").attr("x", -height / 2).attr("y", 14)
    .attr("text-anchor", "middle").attr("fill", "#8b8fa3").attr("font-size", 12).text("Our Model − Steamer");

  g.selectAll("circle").data(data).enter().append("circle")
    .attr("cx", d => x(+d.age)).attr("cy", d => y(d.residual))
    .attr("r", 3.5).attr("fill", d => color(d.residual)).attr("opacity", 0.7)
    .on("mouseenter", function (evt, d) {
      d3.select(this).attr("r", 7).attr("stroke", "#fff").attr("stroke-width", 1);
      showTooltip(evt,
        `<div class="tt-name">${d.name}</div>
         <div class="tt-dim">${d.team} · Age ${Math.round(d.age)}</div>
         <div>Residual: ${d.residual > 0 ? "+" : ""}${d.residual.toFixed(3)}</div>`
      );
    })
    .on("mouseleave", function () { d3.select(this).attr("r", 3.5).attr("stroke", "none"); hideTooltip(); });
}

// ══════════════════════════════════════════════════════════════════
// AGING CURVES PAGE
// ══════════════════════════════════════════════════════════════════
function renderAging() {
  renderAgingAll();
  const select = document.getElementById("aging-stat");
  const render = () => {
    const stat = select.value;
    document.getElementById("aging-detail-title").textContent =
      `${STAT_LABELS[stat]} — HSGP Aging Curve`;
    renderAgingDetail(stat);
  };
  select.addEventListener("change", render);
  render();
}

function renderAgingAll() {
  const { agingCurves } = DATA;
  const container = d3.select("#aging-all");
  container.selectAll("*").remove();

  const width = 700, height = 400;
  const margin = { top: 20, right: 120, bottom: 40, left: 55 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  const svg = container.append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%");
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  const allAges = [], allNorm = [];

  Object.entries(agingCurves).forEach(([stat, curve]) => {
    const maxVal = d3.max(curve, d => d.mean);
    curve.forEach(d => {
      d.normalized = d.mean - maxVal;
      allAges.push(d.age);
      allNorm.push(d.normalized);
    });
  });

  const x = d3.scaleLinear().domain(d3.extent(allAges)).range([0, w]);
  const y = d3.scaleLinear().domain([d3.min(allNorm) * 1.1, d3.max(allNorm) * 1.1 || 0.1]).range([h, 0]);

  g.append("g").attr("class", "axis").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).ticks(10));
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(6));

  // Age 27 reference
  g.append("line").attr("x1", x(27)).attr("y1", 0).attr("x2", x(27)).attr("y2", h)
    .attr("stroke", "#5c6078").attr("stroke-dasharray", "4,4");
  g.append("text").attr("x", x(27) + 4).attr("y", 12).attr("fill", "#5c6078").attr("font-size", 10).text("Age 27");

  svg.append("text").attr("x", width / 2).attr("y", height - 4).attr("text-anchor", "middle")
    .attr("fill", "#8b8fa3").attr("font-size", 12).text("Age");
  svg.append("text").attr("transform", "rotate(-90)").attr("x", -height / 2).attr("y", 14)
    .attr("text-anchor", "middle").attr("fill", "#8b8fa3").attr("font-size", 12).text("Aging Effect (vs Peak)");

  const line = d3.line().x(d => x(d.age)).y(d => y(d.normalized)).curve(d3.curveCatmullRom);

  Object.entries(agingCurves).forEach(([stat, curve]) => {
    g.append("path").datum(curve).attr("d", line)
      .attr("stroke", AGING_COLORS[stat]).attr("stroke-width", 2.5).attr("fill", "none");
  });

  // Legend
  const legend = svg.append("g").attr("transform", `translate(${width - 110}, ${margin.top + 10})`);
  const statNames = { k_rate: "K%", bb_rate: "BB%", hr_rate: "HR Rate", iso: "ISO", babip: "BABIP" };
  Object.entries(agingCurves).forEach(([stat, _], i) => {
    legend.append("rect").attr("x", 0).attr("y", i * 20).attr("width", 14).attr("height", 3)
      .attr("fill", AGING_COLORS[stat]);
    legend.append("text").attr("x", 20).attr("y", i * 20 + 4).attr("fill", "#8b8fa3").attr("font-size", 11)
      .text(statNames[stat]);
  });
}

function renderAgingDetail(stat) {
  const { agingCurves } = DATA;
  const curve = agingCurves[stat];
  if (!curve) { d3.select("#aging-detail").html("<p style='color:#5c6078'>No data</p>"); return; }

  const container = d3.select("#aging-detail");
  container.selectAll("*").remove();

  const width = 700, height = 400;
  const margin = { top: 20, right: 20, bottom: 40, left: 55 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  const svg = container.append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%");
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  const x = d3.scaleLinear().domain(d3.extent(curve, d => d.age)).range([0, w]);

  const hasCI = curve[0].lower != null;
  const allY = hasCI ? curve.flatMap(d => [d.lower, d.upper]) : curve.map(d => d.mean);
  const y = d3.scaleLinear().domain(d3.extent(allY)).nice().range([h, 0]);

  g.append("g").attr("class", "axis").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).ticks(10));
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(6));

  const color = AGING_COLORS[stat];

  if (hasCI) {
    const area = d3.area()
      .x(d => x(d.age)).y0(d => y(d.lower)).y1(d => y(d.upper))
      .curve(d3.curveCatmullRom);
    g.append("path").datum(curve).attr("d", area).attr("fill", color).attr("opacity", 0.15);
  }

  const line = d3.line().x(d => x(d.age)).y(d => y(d.mean)).curve(d3.curveCatmullRom);
  g.append("path").datum(curve).attr("d", line)
    .attr("stroke", color).attr("stroke-width", 2.5).attr("fill", "none");

  // Peak annotation
  const peak = curve.reduce((a, b) => a.mean > b.mean ? a : b);
  g.append("line").attr("x1", x(peak.age)).attr("y1", 0).attr("x2", x(peak.age)).attr("y2", h)
    .attr("stroke", "#f87171").attr("stroke-dasharray", "4,4");
  g.append("text").attr("x", x(peak.age) + 4).attr("y", 12)
    .attr("fill", "#f87171").attr("font-size", 11).text(`Peak: ${peak.age.toFixed(0)}`);

  svg.append("text").attr("x", width / 2).attr("y", height - 4).attr("text-anchor", "middle")
    .attr("fill", "#8b8fa3").attr("font-size", 12).text("Age");
  svg.append("text").attr("transform", "rotate(-90)").attr("x", -height / 2).attr("y", 14)
    .attr("text-anchor", "middle").attr("fill", "#8b8fa3").attr("font-size", 12).text("Aging Effect");
}

// ══════════════════════════════════════════════════════════════════
// LEADERBOARD PAGE
// ══════════════════════════════════════════════════════════════════
function renderLeaderboard() {
  const statSelect = document.getElementById("lb-stat");
  const countSelect = document.getElementById("lb-count");
  const render = () => renderLeaderboardTable(statSelect.value, +countSelect.value);
  statSelect.addEventListener("change", render);
  countSelect.addEventListener("change", render);
  render();
}

function renderLeaderboardTable(stat, count) {
  const { comparison } = DATA;
  const ascending = stat === "k_rate"; // lower K% is better

  // Sort and slice
  const sorted = comparison
    .filter(d => d[`our_${stat}`] !== "" && !isNaN(+d[`our_${stat}`]))
    .sort((a, b) => ascending
      ? +a[`our_${stat}`] - +b[`our_${stat}`]
      : +b[`our_${stat}`] - +a[`our_${stat}`])
    .slice(0, count);

  let html = `<table><thead><tr>
    <th>#</th><th>Player</th><th>Team</th><th>Age</th>`;
  SYSTEMS.forEach(sys => html += `<th style="color:${sys.color}">${sys.label}</th>`);
  html += "</tr></thead><tbody>";

  sorted.forEach((d, i) => {
    html += `<tr><td>${i + 1}</td><td class="name-cell">${d.name}</td>
      <td class="team-cell">${d.team}</td><td>${Math.round(d.age)}</td>`;
    SYSTEMS.forEach(sys => {
      const val = d[`${sys.key}_${stat}`];
      html += `<td>${fmt(val, stat)}</td>`;
    });
    html += "</tr>";
  });
  html += "</tbody></table>";
  document.getElementById("leaderboard-table").innerHTML = html;

  // Rank comparison
  renderRankDelta(stat);
}

function renderRankDelta(stat) {
  const { comparison } = DATA;
  const ascending = stat === "k_rate";

  // Compute ranks
  const withOur = comparison.filter(d => d[`our_${stat}`] !== "" && !isNaN(+d[`our_${stat}`]));
  const withStea = comparison.filter(d => d[`stea_${stat}`] !== "" && !isNaN(+d[`stea_${stat}`]));

  const ourSorted = [...withOur].sort((a, b) => ascending ? +a[`our_${stat}`] - +b[`our_${stat}`] : +b[`our_${stat}`] - +a[`our_${stat}`]);
  const steaSorted = [...withStea].sort((a, b) => ascending ? +a[`stea_${stat}`] - +b[`stea_${stat}`] : +b[`stea_${stat}`] - +a[`stea_${stat}`]);

  const ourRank = {}, steaRank = {};
  ourSorted.forEach((d, i) => ourRank[d.batter] = i + 1);
  steaSorted.forEach((d, i) => steaRank[d.batter] = i + 1);

  const deltas = comparison
    .filter(d => ourRank[d.batter] && steaRank[d.batter])
    .map(d => ({ ...d, ourR: ourRank[d.batter], steaR: steaRank[d.batter], delta: steaRank[d.batter] - ourRank[d.batter] }))
    .sort((a, b) => b.delta - a.delta);

  const higher = deltas.filter(d => d.delta > 0).slice(0, 10);
  const lower = deltas.filter(d => d.delta < 0).slice(0, 10);

  document.getElementById("rank-higher").innerHTML = higher.map(d =>
    `<div class="rank-item"><span class="name">${d.name}</span>
     <span class="tt-dim"> (${d.team})</span>
     — Our #${d.ourR} vs Steamer #${d.steaR}
     <span class="rank-delta-pos">+${d.delta} spots</span></div>`
  ).join("");

  document.getElementById("rank-lower").innerHTML = lower.map(d =>
    `<div class="rank-item"><span class="name">${d.name}</span>
     <span class="tt-dim"> (${d.team})</span>
     — Our #${d.ourR} vs Steamer #${d.steaR}
     <span class="rank-delta-neg">${d.delta} spots</span></div>`
  ).join("");
}

// ══════════════════════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════════════════════
async function init() {
  await loadData();
  initTooltip();
  initNav();
  renderPage("overview");
}

init();
