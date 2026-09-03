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
  const [comparison, ourModel, agingCurves, summary, careerWar, playoffs, accuracy, ros] = await Promise.all([
    d3.json("data/comparison.json"),
    d3.json("data/our_model.json"),
    d3.json("data/aging_curves.json"),
    d3.json("data/summary.json"),
    d3.json("data/career_war.json"),
    d3.json("data/playoff_odds/latest.json").catch(() => null),
    d3.json("data/accuracy/latest.json").catch(() => null),
    // The live rest-of-season projection. Every page that uses it checks for
    // null first: a checkout that has never run build_ros_projections.py still
    // renders, it just shows the preseason numbers alone.
    d3.json("data/projections/latest.json").catch(() => null),
  ]);
  DATA = { comparison, ourModel, agingCurves, summary, careerWar, playoffs, accuracy, ros };
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
    case "playoffs": renderPlayoffs(); break;
    case "accuracy": renderAccuracy(); break;
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
  const sideSel = document.getElementById("player-side");
  const hitters = DATA.comparison.filter(d => d.name && d.name !== "");
  // Pitchers come from the rest-of-season file, which is the only place the
  // site has a pitcher at all; a checkout without it keeps the toggle off.
  const pitchers = rosPitchers().filter(d => d.name);
  if (sideSel && !pitchers.length) sideSel.disabled = true;

  const onPitchers = () => !!(sideSel && sideSel.value === "pitchers");
  const list = () => (onPitchers() ? pitchers : hitters);
  const idOf = d => onPitchers() ? +d.pitcher : +d.batter;
  const show = d => onPitchers() ? renderPitcherCard(d) : renderPlayerCard(d);

  input.addEventListener("input", () => {
    const q = input.value.toLowerCase();
    if (q.length < 2) { sugDiv.classList.remove("visible"); return; }
    const matches = list().filter(d => d.name.toLowerCase().includes(q)).slice(0, 12);
    sugDiv.innerHTML = matches.map(d =>
      `<div class="suggestion-item" data-id="${idOf(d)}">
        <span>${esc(d.name)}</span><span class="team">${esc(d.team || d.team_abbrev || "")}` +
        `${onPitchers() ? " · " + esc(d.role || "") : " · " + Math.round(d.age)}</span>
      </div>`
    ).join("");
    sugDiv.classList.add("visible");
    sugDiv.querySelectorAll(".suggestion-item").forEach(el => {
      el.addEventListener("click", () => {
        const id = +el.dataset.id;
        input.value = el.querySelector("span").textContent;
        sugDiv.classList.remove("visible");
        show(list().find(d => idOf(d) === id));
      });
    });
  });

  input.addEventListener("blur", () => setTimeout(() => sugDiv.classList.remove("visible"), 200));

  // Default: the best player on this side who actually has a rest-of-season
  // projection. Ranking hitters by the preseason number alone lands on
  // whoever is hurt — a card whose headline block says "no projected plate
  // appearances" — so the default skips past them.
  const showDefault = () => {
    if (onPitchers()) {
      const ranked = [...pitchers].sort((a, b) => (+a.fip_ros) - (+b.fip_ros));
      const first = ranked.find(p => +p.bf_ros >= 20) || ranked[0];
      if (!first) return;
      input.value = first.name;
      renderPitcherCard(first);
      return;
    }
    if (!hitters.length) return;
    const byPreseason = [...hitters].sort((a, b) => (b.our_woba || 0) - (a.our_woba || 0));
    const first = byPreseason.find(p => {
      const row = rosPlayer(p.batter);
      return row && +row.pa_ros >= 50;
    }) || byPreseason[0];
    input.value = first.name;
    renderPlayerCard(first);
  };
  if (sideSel) sideSel.addEventListener("change", showDefault);
  showDefault();
}

/** The pitcher card: the rest-of-season block, and nothing else.
 *
 * There is no preseason pitcher projection on this site and no career WAR
 * chart for a pitcher, so this card is deliberately the live block alone
 * rather than a hitter card with most of it blanked out. */
function renderPitcherCard(row) {
  const container = document.getElementById("player-card");
  const careerCard = document.getElementById("career-war-card");
  if (careerCard) careerCard.style.display = "none";
  if (!row) { container.innerHTML = ""; return; }
  const doc = rosDoc();
  const arms = rosPitcherArms();

  let table = '<table class="ros-table"><thead><tr><th>Component</th>';
  arms.forEach(a => {
    table += `<th class="num${a.is_live ? " ros-live-col" : ""}">${esc(a.label)}</th>`;
  });
  table += "</tr></thead><tbody>";
  ROS_PITCHER_STATS.forEach(stat => {
    table += `<tr><td class="name-cell">${stat.label}</td>`;
    arms.forEach(a => {
      table += `<td class="num${a.is_live ? " ros-live-col" : ""}">` +
        `${rosFmt(row[`${stat.key}_rate_${a.key}`])}</td>`;
    });
    table += "</tr>";
  });
  table += "</tbody></table>";

  const tiles = [
    ["BF", rosFmt(row.bf_ros, 0)],
    ["FIP", rosFmt(row.fip_ros, 2)],
    ["K", rosFmt(row.k_ros, 1)],
    ["BB", rosFmt(row.bb_ros, 1)],
    ["HR", rosFmt(row.hr_ros, 1)],
  ].map(([label, value]) =>
    `<div class="ros-tile"><span class="ros-tile-value">${value}</span>
       <span class="ros-tile-label">${label}</span></div>`).join("");

  container.innerHTML = `
    <div class="player-header">
      <h2>${esc(row.name || row.pitcher)}</h2>
      <div class="player-meta">
        <span>${esc(row.team_abbrev || "")}</span>
        <span>${row.role === "SP" ? "Starter" : "Reliever"}</span>
        <span>${rosFmt(row.appearances, 0)} appearances, ` +
          `${rosFmt(row.bf_to_date, 0)} batters faced so far</span>
      </div>
    </div>
    <div class="card ros-card">
      <h3>Rest of season <span class="ros-asof">as of ${esc(row.as_of)}</span></h3>
      ${rosFramingHTML()}
      <div class="ros-tiles">${tiles}</div>
      <div class="table-scroll">${table}</div>
      <p class="method-note">${esc((doc && doc.pitcher_method) || "")}</p>
    </div>`;
  wireROSLinks(container);
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
    <div id="player-ros"></div>
    <div class="grid-2">
      <div class="card"><h3>Component Rates <span class="card-sub">preseason</span></h3><div id="player-components"></div></div>
      <div class="card"><h3>Projection Uncertainty <span class="card-sub">preseason</span></h3><div id="player-uncertainty"></div></div>
    </div>
    <div class="card"><h3>Aggregate Projections <span class="card-sub">preseason, full season</span></h3><div id="player-agg-table"></div></div>
  `;
  container.innerHTML = html;

  // The live rest-of-season projection leads; everything below it is the
  // preseason card this page used to be.
  renderPlayerROS(player);
  renderComponentBars(player);
  renderUncertainty(player);
  renderPlayerAggTable(player);

  // Career WAR chart
  const mlbam = player.batter;
  const career = DATA.careerWar?.[mlbam];
  const careerCard = document.getElementById("career-war-card");
  if (career) {
    careerCard.style.display = "block";
    renderCareerWAR(career);
  } else {
    careerCard.style.display = "none";
  }
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
// REST-OF-SEASON PROJECTION (station A, live)
// ══════════════════════════════════════════════════════════════════
// The number the site leads with. `data/projections/latest.json` is written by
// scripts/build_ros_projections.py: Marcel fed the season to date — the arm
// that wins the intra-season walk-forward — times station B's projected PA.
// The preseason Bayesian components ride along as a labelled comparison
// because they are what this page used to show, and they lose.
//
// Every function here returns early when the file is absent. That is the
// contract: the page predates the projection and must keep working without it.

const ROS_STATS = [
  { key: "k", label: "K%", asc: true },
  { key: "bb", label: "BB%", asc: false },
  { key: "hr", label: "HR/PA", asc: false },
  { key: "iso", label: "ISO", asc: false },
  { key: "babip", label: "BABIP", asc: false },
];
const ROS_FALLBACK_ARMS = [
  { key: "marcel", label: "Live (Marcel + 2026)", is_live: true },
  { key: "bayes", label: "Preseason Bayesian", is_live: false },
  { key: "marcel_preseason", label: "Preseason Marcel", is_live: false },
];

// Short header labels for the leaderboard, where the arm name is prefixed by
// the component and the full "Live (Marcel + 2026)" pushes the third arm off
// a 1280px screen. The player card, which has one column per arm and room to
// spare, uses the long labels from the file.
const ROS_SHORT_LABELS = {
  marcel: "live", bayes: "pre. Bayes", marcel_preseason: "pre. Marcel",
};

// The pitcher block of the same file. Four components, because those are the
// four that cleared the serving gate; the walks-plus-hit-batsmen rate the
// odds model uses is scored in the harness but is not a column here, since a
// column labelled BB% has to mean walks. There is no Bayesian arm — the
// preseason components were only ever fit for hitters.
const ROS_PITCHER_STATS = [
  { key: "k", label: "K%" },
  { key: "bb", label: "BB%" },
  { key: "hr", label: "HR/BF" },
  { key: "babip", label: "BABIP" },
];
const ROS_PITCHER_FALLBACK_ARMS = [
  { key: "marcel", label: "Live (tuned Marcel + 2026)", is_live: true },
  { key: "marcel_preseason", label: "Preseason tuned Marcel", is_live: false },
];

let _rosIndex = null;
let _rosPitcherIndex = null;

function rosDoc() {
  return DATA.ros && Array.isArray(DATA.ros.players) ? DATA.ros : null;
}

function rosArms() {
  const doc = rosDoc();
  return (doc && doc.arms && doc.arms.length) ? doc.arms : ROS_FALLBACK_ARMS;
}

function rosPlayer(batter) {
  const doc = rosDoc();
  if (!doc) return null;
  if (!_rosIndex) {
    _rosIndex = new Map();
    doc.players.forEach(p => _rosIndex.set(+p.batter, p));
  }
  return _rosIndex.get(+batter) || null;
}

/** The pitcher rows, empty in a file written before the pitcher block existed. */
function rosPitchers() {
  const doc = rosDoc();
  return doc && Array.isArray(doc.pitchers) ? doc.pitchers : [];
}

function rosPitcherArms() {
  const doc = rosDoc();
  return (doc && doc.pitcher_arms && doc.pitcher_arms.length)
    ? doc.pitcher_arms : ROS_PITCHER_FALLBACK_ARMS;
}

function rosPitcher(pitcher) {
  if (!_rosPitcherIndex) {
    _rosPitcherIndex = new Map();
    rosPitchers().forEach(p => _rosPitcherIndex.set(+p.pitcher, p));
  }
  return _rosPitcherIndex.get(+pitcher) || null;
}

function rosFmt(v, digits = 3) {
  if (v == null || isNaN(v)) return "—";
  return (+v).toFixed(digits).replace(/^0\./, ".");
}

/** The one-line claim, plus a stale badge when the build could not refresh. */
function rosFramingHTML() {
  const doc = rosDoc();
  if (!doc) return "";
  const badge = doc.stale
    ? '<span class="badge-stale" title="not rebuilt by the latest run">stale</span>'
    : '<span class="badge-fresh">live</span>';
  const reason = doc.stale && doc.stale_reason
    ? `<p class="acc-stale-reason">⚠ ${esc(doc.stale_reason)}</p>` : "";
  return `<p class="ros-framing">${badge} ${esc(doc.framing || "")}
    <a href="#" data-page="accuracy" class="ros-link">Model Accuracy →</a></p>${reason}`;
}

/** Re-arm the sidebar navigation for links rendered after initNav ran. */
function wireROSLinks(root) {
  root.querySelectorAll("a.ros-link").forEach(a => {
    a.addEventListener("click", e => {
      e.preventDefault();
      document.querySelector('#sidebar [data-page="accuracy"]').click();
    });
  });
}

// ─── player card block ───────────────────────────────────────────
function renderPlayerROS(player) {
  const host = document.getElementById("player-ros");
  if (!host) return;
  const row = rosPlayer(player.batter);
  const doc = rosDoc();
  if (!row) {
    // No projected rest-of-season plate appearances: injured, optioned, or the
    // projection has not been built here. Say which rather than showing zeros.
    host.innerHTML = `<div class="card"><h3>Rest of season</h3>
      <p class="method-note">${doc
        ? "No projected rest-of-season plate appearances for this hitter — he is on the injured list, optioned, or off a 40-man roster as of "
          + esc(doc.as_of) + "."
        : "The live rest-of-season projection has not been built in this checkout (run <code>scripts/build_ros_projections.py</code>)."}</p></div>`;
    return;
  }

  const arms = rosArms();
  let table = '<table class="ros-table"><thead><tr><th>Component</th>';
  arms.forEach(a => {
    table += `<th class="num${a.is_live ? " ros-live-col" : ""}">${esc(a.label)}</th>`;
  });
  table += "</tr></thead><tbody>";
  ROS_STATS.forEach(stat => {
    table += `<tr><td class="name-cell">${stat.label}</td>`;
    arms.forEach(a => {
      const v = row[`${stat.key}_rate_${a.key}`];
      table += `<td class="num${a.is_live ? " ros-live-col" : ""}">${rosFmt(v)}</td>`;
    });
    table += "</tr>";
  });
  table += "</tbody></table>";

  const tiles = [
    ["PA", rosFmt(row.pa_ros, 0)],
    ["wOBA", rosFmt(row.woba_ros)],
    ["HR", rosFmt(row.hr_ros, 1)],
    ["BB", rosFmt(row.bb_ros, 1)],
    ["K", rosFmt(row.k_ros, 1)],
  ].map(([label, value]) =>
    `<div class="ros-tile"><span class="ros-tile-value">${value}</span>
       <span class="ros-tile-label">${label}</span></div>`).join("");

  host.innerHTML = `<div class="card ros-card">
    <h3>Rest of season — ${esc(row.team_abbrev || "")} <span class="ros-asof">as of ${esc(row.as_of)}</span></h3>
    ${rosFramingHTML()}
    <div class="ros-tiles">${tiles}</div>
    <div class="table-scroll">${table}</div>
    <p class="method-note">${esc((doc && doc.method) || "")}</p>
  </div>`;
  wireROSLinks(host);
}

// ─── leaderboard block ───────────────────────────────────────────
// One card, two sides. The hitters/pitchers toggle swaps the row source, the
// component menu and the workload column; everything else — the arm columns,
// the framing line, the live-column styling — is shared, because the two
// blocks of the file have the same shape.
function renderROSLeaderboard() {
  const card = document.getElementById("ros-leaderboard-card");
  if (!card) return;
  const doc = rosDoc();
  if (!doc) {
    card.innerHTML = `<h3>Rest of season — live projection</h3>
      <p class="method-note">Not built in this checkout — run
      <code>scripts/build_ros_projections.py</code> to generate
      <code>public/data/projections/latest.json</code>. The preseason
      leaderboard below is unaffected.</p>`;
    return;
  }
  document.getElementById("ros-framing").innerHTML = rosFramingHTML();
  wireROSLinks(card);

  const sideSel = document.getElementById("ros-side");
  const statSel = document.getElementById("ros-component");
  const paSel = document.getElementById("ros-minpa");
  const countSel = document.getElementById("ros-count");
  // A file written before the pitcher block existed has no pitchers to show.
  if (sideSel && !rosPitchers().length) sideSel.disabled = true;

  const side = () => (sideSel && sideSel.value === "pitchers") ? "pitchers" : "hitters";
  const fillStats = () => {
    const stats = side() === "pitchers" ? ROS_PITCHER_STATS : ROS_STATS;
    const keep = statSel.value;
    statSel.innerHTML = stats.map(s =>
      `<option value="${s.key}">${esc(s.label)}</option>`).join("");
    if (stats.some(s => s.key === keep)) statSel.value = keep;
  };
  const draw = () => {
    document.getElementById("ros-minpa-label").textContent =
      side() === "pitchers" ? "Min BF: " : "Min PA: ";
    document.getElementById("ros-method").textContent =
      (side() === "pitchers" ? doc.pitcher_method : doc.method) || "";
    renderROSLeaderboardTable(side(), statSel.value, +paSel.value, +countSel.value);
  };
  if (sideSel) sideSel.addEventListener("change", () => { fillStats(); draw(); });
  [statSel, paSel, countSel].forEach(el => el.addEventListener("change", draw));
  fillStats();
  draw();
}

function renderROSLeaderboardTable(side, statKey, minWork, count) {
  const doc = rosDoc();
  const pitchers = side === "pitchers";
  const stats = pitchers ? ROS_PITCHER_STATS : ROS_STATS;
  const stat = stats.find(s => s.key === statKey) || stats[0];
  const arms = pitchers ? rosPitcherArms() : rosArms();
  // Hitters rank by wOBA descending, pitchers by FIP ascending: on both sides
  // the top of the table is the better player.
  const workKey = pitchers ? "bf_ros" : "pa_ros";
  const summaryKey = pitchers ? "fip_ros" : "woba_ros";
  const rows = (pitchers ? rosPitchers() : doc.players)
    .filter(p => p[summaryKey] != null && +p[workKey] >= minWork)
    .sort((a, b) => pitchers ? (+a.fip_ros) - (+b.fip_ros)
                             : (+b.woba_ros) - (+a.woba_ros))
    .slice(0, count);

  let h = '<table class="acc-table ros-table"><thead><tr>' +
    "<th>#</th><th>Player</th><th>Team</th>" +
    (pitchers ? '<th>Role</th><th class="num">BF</th>' : '<th class="num">PA</th>') +
    '<th class="num">K</th><th class="num">BB</th><th class="num">HR</th>' +
    `<th class="num ros-live-col">${pitchers ? "FIP" : "wOBA"}</th>`;
  arms.forEach(a => {
    const label = ROS_SHORT_LABELS[a.key] || a.label;
    h += `<th class="num${a.is_live ? " ros-live-col" : ""}" title="${esc(a.label)}">` +
      `${esc(stat.label)} ${esc(label)}</th>`;
  });
  h += "</tr></thead><tbody>";

  rows.forEach((p, i) => {
    h += `<tr><td>${i + 1}</td>` +
      `<td class="name-cell">${esc(p.name || p.pitcher || p.batter)}</td>` +
      `<td class="team-cell">${esc(p.team_abbrev || "")}</td>` +
      (pitchers ? `<td class="team-cell">${esc(p.role || "")}</td>` : "") +
      `<td class="num">${rosFmt(p[workKey], 0)}</td>` +
      `<td class="num">${rosFmt(p.k_ros, 1)}</td>` +
      `<td class="num">${rosFmt(p.bb_ros, 1)}</td>` +
      `<td class="num">${rosFmt(p.hr_ros, 1)}</td>` +
      `<td class="num ros-live-col">` +
      `${pitchers ? rosFmt(p.fip_ros, 2) : rosFmt(p.woba_ros)}</td>`;
    arms.forEach(a => {
      h += `<td class="num${a.is_live ? " ros-live-col" : ""}">` +
        `${rosFmt(p[`${stat.key}_rate_${a.key}`])}</td>`;
    });
    h += "</tr>";
  });
  h += "</tbody></table>";
  document.getElementById("ros-leaderboard-table").innerHTML = h;
  const total = pitchers ? (doc.n_pitchers || rosPitchers().length) : doc.n_hitters;
  document.getElementById("ros-count-note").textContent =
    `${rows.length} of ${total} projected ${pitchers ? "pitchers" : "hitters"} ` +
    `shown (at least ${minWork} projected ` +
    `${pitchers ? "batters faced" : "plate appearances"}), ranked by projected ` +
    `rest-of-season ${pitchers ? "FIP" : "wOBA"}.`;
}


// ══════════════════════════════════════════════════════════════════
// CAREER WAR CHART — with Bayesian uncertainty bands
// ══════════════════════════════════════════════════════════════════
const COMP_COLORS = {
  fitted: "#f59e0b",
  steamer: "#ef4444",
  zips: "#a855f7",
  depthcharts: "#06b6d4",
};

function renderCareerWAR(career) {
  const container = d3.select("#career-war-chart");
  container.selectAll("*").remove();

  const hist = career.historical || [];
  const proj = career.projected || [];
  const fitted = career.fitted || [];
  const comparisons = career.comparisons || {};

  if (hist.length === 0) return;

  // Detect new format (war_p50) vs legacy (war)
  const hasUncertainty = proj.length > 0 && proj[0].war_p50 !== undefined;
  const hasFitted = fitted.length > 0;
  const hasComparisons = Object.keys(comparisons).length > 0;

  // ── Toggle controls ──────────────────────────────────────────────
  const toggleDiv = container.append("div")
    .attr("class", "career-toggles")
    .style("display", "flex").style("gap", "12px").style("margin-bottom", "8px")
    .style("font-size", "12px").style("flex-wrap", "wrap").style("align-items", "center");

  const toggles = [
    { key: "fitted", label: "Model Fit", color: COMP_COLORS.fitted, available: hasFitted, defaultOn: true },
    { key: "steamer", label: "Steamer", color: COMP_COLORS.steamer, available: !!comparisons.steamer, defaultOn: false },
    { key: "zips", label: "ZiPS", color: COMP_COLORS.zips, available: !!comparisons.zips, defaultOn: false },
    { key: "depthcharts", label: "Depth Charts", color: COMP_COLORS.depthcharts, available: !!comparisons.depthcharts, defaultOn: false },
  ];

  toggles.filter(t => t.available).forEach(t => {
    const label = toggleDiv.append("label")
      .style("cursor", "pointer").style("display", "flex").style("align-items", "center").style("gap", "4px");
    label.append("input")
      .attr("type", "checkbox")
      .attr("data-sys", t.key)
      .property("checked", t.defaultOn);
    label.append("span")
      .style("color", t.color)
      .text(`● ${t.label}`);
  });

  const width = 750, height = 380;
  const margin = { top: 30, right: 30, bottom: 45, left: 55 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  const svg = container.append("svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("width", "100%");
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  // Get WAR value for a projected point (handles both formats)
  const projWar = d => hasUncertainty ? d.war_p50 : d.war;

  // Combine data for scale computation
  const allYears = [...hist.map(d => d.year), ...proj.map(d => d.year)];

  // Bridge point: last historical year connects to first projection
  const lastHist = hist[hist.length - 1];

  // Scales
  const x = d3.scaleLinear()
    .domain([d3.min(allYears) - 0.5, d3.max(allYears) + 0.5])
    .range([0, w]);

  // Y domain: consider uncertainty bands and comparison values
  const histWars = hist.map(d => d.war);
  const projWars = hasUncertainty
    ? [...proj.map(d => d.war_p5), ...proj.map(d => d.war_p95)]
    : proj.map(d => d.war);
  const compWars = Object.values(comparisons).map(c => c.war);
  const fittedWars = fitted.map(d => d.war);
  const allWar = [...histWars, ...projWars, ...compWars, ...fittedWars];
  const yMin = Math.min(0, d3.min(allWar) - 0.5);
  const yMax = d3.max(allWar) + 1;
  const y = d3.scaleLinear().domain([yMin, yMax]).range([h, 0]);

  // Grid lines
  g.selectAll(".grid-line")
    .data(y.ticks(6))
    .enter().append("line")
    .attr("x1", 0).attr("x2", w)
    .attr("y1", d => y(d)).attr("y2", d => y(d))
    .attr("stroke", "#1e2130").attr("stroke-width", 1);

  // Zero line
  g.append("line")
    .attr("x1", 0).attr("y1", y(0)).attr("x2", w).attr("y2", y(0))
    .attr("stroke", "#3a3d4a").attr("stroke-width", 1).attr("stroke-dasharray", "4,4");

  // Axes
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${h})`)
    .call(d3.axisBottom(x).ticks(allYears.length).tickFormat(d3.format("d")));
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(6));

  // Axis labels
  svg.append("text").attr("x", width / 2).attr("y", height - 4)
    .attr("text-anchor", "middle").attr("fill", "#8b8fa3").attr("font-size", 12).text("Season");
  svg.append("text").attr("transform", "rotate(-90)").attr("x", -height / 2).attr("y", 14)
    .attr("text-anchor", "middle").attr("fill", "#8b8fa3").attr("font-size", 12).text("WAR");

  // Projection zone background
  if (proj.length > 0) {
    const projStart = x(proj[0].year - 0.5);
    g.append("rect")
      .attr("x", projStart).attr("y", 0)
      .attr("width", w - projStart).attr("height", h)
      .attr("fill", "#4f8ff7").attr("opacity", 0.04);

    g.append("text")
      .attr("x", projStart + 8).attr("y", 16)
      .attr("fill", "#4f8ff7").attr("font-size", 11).attr("opacity", 0.6)
      .text("Projected (Bayesian)");
  }

  // ── Fitted model line (behind bars) ────────────────────────────────
  const gFitted = g.append("g").attr("class", "layer-fitted")
    .style("display", hasFitted ? null : "none");

  if (hasFitted) {
    const fittedLine = d3.line()
      .x(d => x(d.year)).y(d => y(d.war))
      .curve(d3.curveMonotoneX);

    gFitted.append("path").datum(fitted)
      .attr("d", fittedLine)
      .attr("stroke", COMP_COLORS.fitted).attr("stroke-width", 1.5)
      .attr("fill", "none").attr("opacity", 0.6)
      .attr("stroke-dasharray", "4,3");

    gFitted.selectAll(".dot-fitted").data(fitted).enter().append("circle")
      .attr("cx", d => x(d.year)).attr("cy", d => y(d.war))
      .attr("r", 3).attr("fill", COMP_COLORS.fitted).attr("opacity", 0.7)
      .on("mouseenter", function (evt, d) {
        d3.select(this).attr("r", 6).attr("opacity", 1);
        const actual = hist.find(h => h.year === d.year);
        showTooltip(evt,
          `<div class="tt-name">Model Fit — ${d.year}</div>
           <div class="tt-dim">Age ${d.age}</div>
           <div><strong>${d.war} WAR</strong> fitted (wOBA ${d.woba})</div>
           ${actual ? `<div style="color:#34d399">Actual: ${actual.war} WAR</div>` : ""}`
        );
      })
      .on("mouseleave", function () { d3.select(this).attr("r", 3).attr("opacity", 0.7); hideTooltip(); });
  }

  // Historical line
  const histLine = d3.line()
    .x(d => x(d.year)).y(d => y(d.war))
    .curve(d3.curveMonotoneX);

  g.append("path").datum(hist)
    .attr("d", histLine)
    .attr("stroke", "#34d399").attr("stroke-width", 2.5).attr("fill", "none");

  // Historical bars
  const barWidth = Math.min(w / allYears.length * 0.6, 24);
  g.selectAll(".bar-hist").data(hist).enter().append("rect")
    .attr("x", d => x(d.year) - barWidth / 2)
    .attr("y", d => d.war >= 0 ? y(d.war) : y(0))
    .attr("width", barWidth)
    .attr("height", d => Math.abs(y(0) - y(d.war)))
    .attr("fill", "#34d399").attr("opacity", 0.3).attr("rx", 3);

  // ── Uncertainty bands (new) ──────────────────────────────────────
  if (proj.length > 0 && hasUncertainty) {
    // 90% band (p5–p95) — lightest
    const area95 = d3.area()
      .x(d => x(d.year))
      .y0(d => y(d.war_p5))
      .y1(d => y(d.war_p95))
      .curve(d3.curveMonotoneX);
    g.append("path").datum(proj)
      .attr("d", area95)
      .attr("fill", "#4f8ff7").attr("opacity", 0.08);

    // 80% band (p10–p90)
    const area80 = d3.area()
      .x(d => x(d.year))
      .y0(d => y(d.war_p10))
      .y1(d => y(d.war_p90))
      .curve(d3.curveMonotoneX);
    g.append("path").datum(proj)
      .attr("d", area80)
      .attr("fill", "#4f8ff7").attr("opacity", 0.12);

    // 50% band (p25–p75) — darkest
    const area50 = d3.area()
      .x(d => x(d.year))
      .y0(d => y(d.war_p25))
      .y1(d => y(d.war_p75))
      .curve(d3.curveMonotoneX);
    g.append("path").datum(proj)
      .attr("d", area50)
      .attr("fill", "#4f8ff7").attr("opacity", 0.18);

    // Median line (dashed)
    const projWithBridge = [{ ...lastHist, war_p50: lastHist.war }, ...proj];
    const projLine = d3.line()
      .x(d => x(d.year)).y(d => y(d.war_p50))
      .curve(d3.curveMonotoneX);

    g.append("path").datum(projWithBridge)
      .attr("d", projLine)
      .attr("stroke", "#4f8ff7").attr("stroke-width", 2.5)
      .attr("fill", "none").attr("stroke-dasharray", "8,4");

  } else if (proj.length > 0) {
    // Legacy: single projected line (no uncertainty)
    const projWithBridge = [lastHist, ...proj];
    const projLine = d3.line()
      .x(d => x(d.year)).y(d => y(d.war))
      .curve(d3.curveMonotoneX);

    g.append("path").datum(projWithBridge)
      .attr("d", projLine)
      .attr("stroke", "#4f8ff7").attr("stroke-width", 2.5)
      .attr("fill", "none").attr("stroke-dasharray", "8,4");

    // Projected bars (legacy only)
    g.selectAll(".bar-proj").data(proj).enter().append("rect")
      .attr("x", d => x(d.year) - barWidth / 2)
      .attr("y", d => d.war >= 0 ? y(d.war) : y(0))
      .attr("width", barWidth)
      .attr("height", d => Math.abs(y(0) - y(d.war)))
      .attr("fill", "#4f8ff7").attr("opacity", 0.25).attr("rx", 3);
  }

  // ── Comparison system overlays ──────────────────────────────────────
  const compSystems = [
    { key: "steamer", label: "Steamer" },
    { key: "zips", label: "ZiPS" },
    { key: "depthcharts", label: "Depth Charts" },
  ];

  compSystems.forEach(sys => {
    const data = comparisons[sys.key];
    if (!data) return;

    const color = COMP_COLORS[sys.key];
    const gComp = g.append("g").attr("class", `layer-${sys.key}`)
      .style("display", "none"); // off by default

    // Determine projection zone x-range
    const projYearMin = proj.length > 0 ? proj[0].year : (lastHist ? lastHist.year + 1 : 2026);
    const projYearMax = proj.length > 0 ? proj[proj.length - 1].year : projYearMin;

    // Horizontal dashed line across projection zone
    gComp.append("line")
      .attr("x1", x(projYearMin - 0.3)).attr("x2", x(projYearMax + 0.3))
      .attr("y1", y(data.war)).attr("y2", y(data.war))
      .attr("stroke", color).attr("stroke-width", 1.5)
      .attr("stroke-dasharray", "6,3").attr("opacity", 0.7);

    // Dot at 2026
    const dotYear = proj.length > 0 ? proj[0].year : 2026;
    gComp.append("circle")
      .attr("cx", x(dotYear)).attr("cy", y(data.war))
      .attr("r", 5).attr("fill", color).attr("stroke", "#0f1117").attr("stroke-width", 1)
      .on("mouseenter", function (evt) {
        d3.select(this).attr("r", 8);
        showTooltip(evt,
          `<div class="tt-name">${sys.label} — ${dotYear}</div>
           <div><strong>${data.war} WAR</strong> · wOBA ${data.woba} · ${data.pa} PA</div>`
        );
      })
      .on("mouseleave", function () { d3.select(this).attr("r", 5); hideTooltip(); });

    // Label
    gComp.append("text")
      .attr("x", x(projYearMax) + 5).attr("y", y(data.war) + 4)
      .attr("fill", color).attr("font-size", 10).attr("font-weight", 600)
      .text(`${sys.label} ${data.war}`);
  });

  // Historical dots
  g.selectAll(".dot-hist").data(hist).enter().append("circle")
    .attr("cx", d => x(d.year)).attr("cy", d => y(d.war))
    .attr("r", 5).attr("fill", "#34d399").attr("stroke", "#0f1117").attr("stroke-width", 1.5)
    .on("mouseenter", function (evt, d) {
      d3.select(this).attr("r", 8);
      showTooltip(evt,
        `<div class="tt-name">${career.name} — ${d.year}</div>
         <div class="tt-dim">${d.team || ""} · Age ${d.age}</div>
         <div><strong>${d.war} WAR</strong> · ${d.pa} PA</div>
         ${d.woba ? `<div>wOBA ${d.woba} · wRC+ ${Math.round(d.wrc_plus || 0)} · Off ${d.off}</div>` : ""}`
      );
    })
    .on("mouseleave", function () { d3.select(this).attr("r", 5); hideTooltip(); });

  // Projected dots
  if (proj.length > 0) {
    g.selectAll(".dot-proj").data(proj).enter().append("circle")
      .attr("cx", d => x(d.year)).attr("cy", d => y(hasUncertainty ? d.war_p50 : d.war))
      .attr("r", 5).attr("fill", "#4f8ff7").attr("stroke", "#0f1117").attr("stroke-width", 1.5)
      .on("mouseenter", function (evt, d) {
        d3.select(this).attr("r", 8);
        if (hasUncertainty) {
          showTooltip(evt,
            `<div class="tt-name">${career.name} — ${d.year}</div>
             <div class="tt-dim">Projected · Age ${d.age}</div>
             <div><strong>${d.war_p50} WAR</strong> <span style="color:#8b8fa3">(median)</span></div>
             <div style="font-size:12px;color:#8b8fa3">
               90% CI: ${d.war_p5}–${d.war_p95} · 80% CI: ${d.war_p10}–${d.war_p90}
             </div>
             ${d.woba_p50 ? `<div style="margin-top:4px">wOBA ${d.woba_p50} · wRC+ ${Math.round(d.wrc_plus_p50 || 0)}</div>` : ""}`
          );
        } else {
          showTooltip(evt,
            `<div class="tt-name">${career.name} — ${d.year}</div>
             <div class="tt-dim">Projected · Age ${d.age}</div>
             <div><strong>${d.war} WAR</strong></div>`
          );
        }
      })
      .on("mouseleave", function () { d3.select(this).attr("r", 5); hideTooltip(); });
  }

  // WAR labels on dots
  hist.forEach(d => {
    if (Math.abs(d.war) >= 1.0) {
      g.append("text")
        .attr("x", x(d.year)).attr("y", y(d.war) - 10)
        .attr("text-anchor", "middle")
        .attr("fill", "#34d399")
        .attr("font-size", 10).attr("font-weight", 600)
        .text(d.war.toFixed(1));
    }
  });
  proj.forEach(d => {
    const warVal = hasUncertainty ? d.war_p50 : d.war;
    g.append("text")
      .attr("x", x(d.year)).attr("y", y(warVal) - 10)
      .attr("text-anchor", "middle")
      .attr("fill", "#4f8ff7")
      .attr("font-size", 10).attr("font-weight", 600)
      .text(warVal.toFixed(1));
  });

  // Legend
  const legend = svg.append("g").attr("transform", `translate(${margin.left + 10}, ${margin.top - 15})`);
  // Actual
  legend.append("rect").attr("x", 0).attr("y", 0).attr("width", 14).attr("height", 3).attr("fill", "#34d399");
  legend.append("text").attr("x", 18).attr("y", 4).attr("fill", "#8b8fa3").attr("font-size", 11).text("Actual");
  // Projected
  legend.append("rect").attr("x", 80).attr("y", 0).attr("width", 14).attr("height", 3).attr("fill", "#4f8ff7");
  legend.append("text").attr("x", 98).attr("y", 4).attr("fill", "#8b8fa3").attr("font-size", 11).text("Projected");
  if (hasUncertainty) {
    // Uncertainty band legend
    legend.append("rect").attr("x", 180).attr("y", -3).attr("width", 14).attr("height", 10)
      .attr("fill", "#4f8ff7").attr("opacity", 0.18).attr("rx", 2);
    legend.append("text").attr("x", 198).attr("y", 4).attr("fill", "#8b8fa3").attr("font-size", 11).text("50/80/90% CI");
  }
  // Career total
  const totalActual = d3.sum(hist, d => d.war);
  const totalProj = hasUncertainty ? d3.sum(proj, d => d.war_p50) : d3.sum(proj, d => d.war);
  legend.append("text").attr("x", hasUncertainty ? 290 : 190).attr("y", 4).attr("fill", "#5c6078").attr("font-size", 11)
    .text(`Career: ${totalActual.toFixed(1)} actual + ${totalProj.toFixed(1)} proj = ${(totalActual + totalProj).toFixed(1)} total`);

  // ── Wire up toggle event listeners ──────────────────────────────────
  toggleDiv.selectAll("input[data-sys]").on("change", function () {
    const sys = this.dataset.sys;
    const visible = this.checked;
    g.select(`.layer-${sys}`).style("display", visible ? null : "none");
  });
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
  renderROSLeaderboard();
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
// PLAYOFF ODDS (Phase 2/3) — reads data/playoff_odds/latest.json
// ══════════════════════════════════════════════════════════════════
const ODDS_COLS = [
  { key: "p_playoffs", label: "Playoffs" },
  { key: "p_division", label: "Div" },
  { key: "p_bye", label: "Bye" },
  { key: "p_pennant", label: "Pennant" },
  { key: "p_ws", label: "WS" },
];
const DIV_SHORT = { "AL East": "East", "AL Central": "Central", "AL West": "West",
                    "NL East": "East", "NL Central": "Central", "NL West": "West" };
let oddsSort = { key: "p_playoffs", dir: -1 };

function probCell(p) {
  const pct = p * 100;
  const txt = pct >= 99.95 ? "100" : pct < 0.05 ? "<0.1" : pct >= 10 ? pct.toFixed(0) : pct.toFixed(1);
  const bg = `rgba(79,143,247,${(0.05 + 0.55 * p).toFixed(2)})`;
  return `<td class="prob" style="background:${bg}">${txt}</td>`;
}

function renderPlayoffs() {
  const d = DATA.playoffs;
  if (!d) {
    document.getElementById("playoffs-subtitle").innerHTML =
      '<span class="stale-warning">No odds snapshot found — run scripts/run_playoff_odds.py.</span>';
    return;
  }
  const asOf = new Date(d.as_of + "T12:00:00");
  const ageDays = Math.floor((Date.now() - new Date(d.generated_at)) / 86400000);
  const stale = ageDays >= 2 ? ` <span class="stale-warning">⚠ ${ageDays} days old</span>` : "";
  document.getElementById("playoffs-title").textContent = `${d.season} Playoff Odds`;
  document.getElementById("playoffs-subtitle").innerHTML =
    `Through games of <strong>${asOf.toLocaleDateString("en-US", { month: "long", day: "numeric" })}</strong>` +
    ` · ${d.n_sims.toLocaleString()} simulated seasons · updated ${new Date(d.generated_at).toLocaleString()}${stale}`;

  const inPlay = d.teams.filter(t => t.p_playoffs > 0.0005 && t.p_playoffs < 0.9995).length;
  document.getElementById("playoffs-metrics").innerHTML = [
    { label: "Games Remaining", value: d.games_remaining },
    { label: "Teams Still In Play", value: inPlay },
    { label: "Home-Field Edge", value: (d.hfa * 100).toFixed(1) + "%" },
    { label: "Simulations", value: d.n_sims.toLocaleString() },
  ].map(m => `<div class="metric-card"><div class="label">${m.label}</div><div class="value">${m.value}</div></div>`).join("");

  renderOddsTable("playoffs-table-al", d.teams.filter(t => t.league_id === 103));
  renderOddsTable("playoffs-table-nl", d.teams.filter(t => t.league_id === 104));
  renderBracket(d.teams);
  document.getElementById("playoffs-method").textContent = d.method +
    ". Pythagenpat converts each club's run rates to a talent win%, and every remaining game whose starters " +
    "nobody has announced yet is drawn with log5 plus a home-field multiplier; ties are broken by head-to-head, " +
    "intradivision, then intraleague-second-half record. " +
    "Snapshots are archived daily and never overwritten. " +
    // The measured framing, from docs/team-projection-backtest.md: 7,470
    // club-projections an arm over 2015-2025. Keep this honest about where
    // the edge is, because it is not here in September.
    "Scored walk-forward on 2015-2025, the projected-wins column beats a .500 extrapolation of the current " +
    "record by 1.31 wins of mean absolute error (2.00 in April, 0.14 in the last fortnight). The playoff, " +
    "division, pennant and World Series probabilities beat that same extrapolation early and stop beating it " +
    "around the start of August: past that point the standings have decided the season and this board is " +
    "mostly arithmetic. See the Model Accuracy page.";
}

function renderOddsTable(elId, teams) {
  const rows = [...teams].sort((a, b) => {
    const diff = (a[oddsSort.key] - b[oddsSort.key]) * oddsSort.dir;
    return diff !== 0 ? diff : b.mean_wins - a.mean_wins;
  });
  let h = '<table class="odds-table"><thead><tr><th>Team</th><th>Div</th><th>W-L</th>' +
    `<th class="sortable ${oddsSort.key === "mean_wins" ? "sorted" : ""}" data-sort="mean_wins">Proj W</th>` +
    "<th>90% range</th>";
  ODDS_COLS.forEach(c => {
    h += `<th class="sortable ${oddsSort.key === c.key ? "sorted" : ""}" data-sort="${c.key}">${c.label}</th>`;
  });
  h += "</tr></thead><tbody>";
  rows.forEach(t => {
    const cls = t.p_playoffs >= 0.9995 ? "clinched" : t.p_playoffs < 0.0005 ? "eliminated" : "";
    h += `<tr class="${cls}"><td class="team-cell">${t.abbrev}</td><td>${DIV_SHORT[t.division] || t.division}</td>` +
      `<td class="wl">${t.wins}-${t.losses}</td><td class="prob">${t.mean_wins.toFixed(1)}</td>` +
      `<td class="range">${Math.round(t.wins_p5)}–${Math.round(t.wins_p95)}</td>`;
    ODDS_COLS.forEach(c => { h += probCell(t[c.key]); });
    h += "</tr>";
  });
  h += "</tbody></table>";
  const el = document.getElementById(elId);
  el.innerHTML = h;
  el.querySelectorAll("th.sortable").forEach(th => th.addEventListener("click", () => {
    const key = th.dataset.sort;
    oddsSort = { key, dir: oddsSort.key === key ? -oddsSort.dir : -1 };
    renderOddsTable("playoffs-table-al", DATA.playoffs.teams.filter(t => t.league_id === 103));
    renderOddsTable("playoffs-table-nl", DATA.playoffs.teams.filter(t => t.league_id === 104));
  }));
}

function currentSeeds(teams) {
  // If the season ended today: division leaders by win%, then three best others.
  const byPct = [...teams].sort((a, b) =>
    (b.wins / (b.wins + b.losses)) - (a.wins / (a.wins + a.losses)) || b.wins - a.wins);
  const winners = [], seen = new Set();
  byPct.forEach(t => { if (!seen.has(t.division)) { seen.add(t.division); winners.push(t); } });
  const wild = byPct.filter(t => !winners.includes(t)).slice(0, 3);
  return [...winners, ...wild];
}

function renderBracket(teams) {
  const html = [[103, "American League"], [104, "National League"]].map(([lg, name]) => {
    const s = currentSeeds(teams.filter(t => t.league_id === lg));
    const tag = (i) => `<span class="seed">${i + 1}</span> ${s[i].abbrev} <span class="seed">(${s[i].wins}-${s[i].losses})</span>`;
    return `<div class="bracket"><h4>${name}</h4>` +
      `<div>${tag(0)} <span class="bye">bye</span> → faces winner of 4/5</div>` +
      `<div>${tag(1)} <span class="bye">bye</span> → faces winner of 3/6</div>` +
      `<div>${tag(2)} vs ${tag(5)} &nbsp;(best-of-3 at ${s[2].abbrev})</div>` +
      `<div>${tag(3)} vs ${tag(4)} &nbsp;(best-of-3 at ${s[3].abbrev})</div></div>`;
  }).join("");
  document.getElementById("playoffs-bracket").innerHTML = html;
}


// ══════════════════════════════════════════════════════════════════
// MODEL ACCURACY (station H) — reads data/accuracy/latest.json
//
// Every number on this page comes out of that file, which
// scripts/build_accuracy_json.py generates from the scoring scripts.
// Nothing here is hard-coded: this renderer only formats and labels.
// ══════════════════════════════════════════════════════════════════
const ACCURACY_ORDER = ["ros_backtest", "pitcher_ros_backtest", "pitcher_workload",
                        "components", "contact_quality",
                        "game_odds", "team_backtest", "playoff_odds_control"];

function esc(v) {
  return String(v == null ? "" : v)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Formatting is per column *type*, declared by the builder — the page never
// decides how many digits a metric deserves on its own.
function accFmt(v, type) {
  if (v == null || (typeof v === "number" && !isFinite(v))) return "—";
  switch (type) {
    case "rate": return (+v).toFixed(4).replace(/^0\./, ".");
    case "score": return (+v).toFixed(4);
    case "prob": return (+v).toFixed(3);
    case "gap": return (+v).toFixed(2);
    case "rank_value": return (+v).toFixed(1);
    case "rank": return String(v);
    default: return esc(v);
  }
}

const LOWER_IS_BETTER = new Set(["rate", "score"]);

function accCellValue(row, col) {
  if (col.key in row) return row[col.key];
  return row.metrics ? row.metrics[col.key] : undefined;
}

function accuracyTable(section) {
  const cols = section.columns || [];
  const rows = section.rows || [];
  if (!cols.length || !rows.length) return '<p class="method-note">No rows in this section.</p>';

  // Best value per lower-is-better column, so the winner is visible at a glance.
  // A section whose rows are not comparable top to bottom (the rest-of-season
  // table: a later cutoff scores a shorter, noisier window) sets
  // highlight_best:false and marks the winners itself, per row, in `best`.
  const best = {};
  if (section.highlight_best !== false) {
    cols.forEach(c => {
      if (!LOWER_IS_BETTER.has(c.type)) return;
      const vals = rows.map(r => accCellValue(r, c)).filter(v => typeof v === "number");
      if (vals.length) best[c.key] = Math.min(...vals);
    });
  }

  let h = '<table class="acc-table"><thead><tr>';
  cols.forEach(c => {
    h += `<th class="${c.type === "text" ? "" : "num"}">${esc(c.label)}</th>`;
  });
  h += "</tr></thead><tbody>";
  // Tags belong on the row's *name*, which is the first text column — a second
  // text column (the rest-of-season table's cutoff) must not repeat them.
  const nameKey = (cols.find(c => c.type === "text") || {}).key;
  rows.forEach(r => {
    const cls = [r.is_market ? "acc-market-row" : "", r.is_control ? "acc-control-row" : "",
                 r.is_ours ? "acc-ours-row" : "",
                 r.is_production ? "acc-live-row" : ""].filter(Boolean).join(" ");
    h += `<tr class="${cls}">`;
    cols.forEach(c => {
      const v = accCellValue(r, c);
      if (c.type === "text") {
        const tags = c.key !== nameKey ? "" :
          [r.is_market ? '<span class="acc-tag acc-tag-market">market</span>' : "",
           r.is_control ? '<span class="acc-tag acc-tag-control">control</span>' : "",
           r.is_ours ? '<span class="acc-tag acc-tag-ours">ours</span>' : "",
           r.is_production ? '<span class="acc-tag acc-tag-prod">live</span>' : ""].join("");
        // A sampled arm carries the scale it was actually fitted at, on the
        // row. A reduced fit is evidence about a reduced fit, and a reader
        // who never opens the JSON has to be able to see which one this is.
        const scale = (c.key === nameKey && r.scale)
          ? `<span class="acc-scale">${esc(r.scale)}</span>` : "";
        h += `<td class="name-cell">${esc(v)}${tags}${scale}</td>`;
      } else {
        const isBest = (c.key in best && v === best[c.key])
          || (Array.isArray(r.best) && r.best.includes(c.key));
        h += `<td class="num${isBest ? " best" : ""}">${accFmt(v, c.type)}</td>`;
      }
    });
    h += "</tr>";
  });
  return h + "</tbody></table>";
}

function accuracySection(name, section) {
  const badge = section.stale
    ? '<span class="badge-stale" title="not regenerated by the latest build">stale</span>'
    : '<span class="badge-fresh">fresh</span>';
  const meta = [
    section.as_of ? `as of <strong>${esc(section.as_of)}</strong>` : null,
    section.n != null ? `n = ${esc(section.n)} ${esc(section.n_label || "")}`.trim() : null,
    section.source ? `source: <code>${esc(section.source)}</code>` : null,
  ].filter(Boolean).join(" · ");
  const notes = (section.notes || []).map(n => `<li>${esc(n)}</li>`).join("");
  return `<div class="card acc-card" id="acc-${esc(name)}">
    <h3>${esc(section.title || name)} ${badge}</h3>
    <p class="acc-framing">${esc(section.framing || "")}</p>
    ${section.stale && section.stale_reason
      ? `<p class="acc-stale-reason">⚠ ${esc(section.stale_reason)}</p>` : ""}
    <p class="acc-meta">${meta}</p>
    <div class="table-scroll">${accuracyTable(section)}</div>
    ${notes ? `<ul class="acc-notes">${notes}</ul>` : ""}
  </div>`;
}

function renderAccuracy() {
  const d = DATA.accuracy;
  const host = document.getElementById("accuracy-sections");
  if (!d) {
    document.getElementById("accuracy-subtitle").innerHTML =
      '<span class="stale-warning">No accuracy snapshot found — run ' +
      'scripts/build_accuracy_json.py.</span>';
    host.innerHTML = "";
    return;
  }
  const sha = d.git_sha ? ` · built from <code>${esc(d.git_sha.slice(0, 7))}</code>` : "";
  document.getElementById("accuracy-subtitle").innerHTML =
    `${esc(d.subtitle || "")} Generated ${new Date(d.generated_at).toLocaleString()}${sha}.`;

  const names = ACCURACY_ORDER.filter(n => d.sections && d.sections[n])
    .concat(Object.keys(d.sections || {}).filter(n => !ACCURACY_ORDER.includes(n)));
  host.innerHTML = names.map(n => accuracySection(n, d.sections[n])).join("");

  document.getElementById("accuracy-glossary").innerHTML =
    (d.meta?.glossary || []).map(g =>
      `<p class="acc-glossary"><strong>${esc(g.term)}</strong> — ${esc(g.text)}</p>`).join("");

  const status = d.meta?.status || [];
  let h = '<table class="acc-table"><thead><tr><th>Section</th><th>State</th>' +
    "<th>As of</th><th>Source</th><th>Why</th></tr></thead><tbody>";
  status.forEach(s => {
    h += `<tr><td class="name-cell">${esc(s.section)}</td>` +
      `<td>${s.fresh ? '<span class="badge-fresh">fresh</span>'
                     : '<span class="badge-stale">stale</span>'}</td>` +
      `<td>${esc(s.as_of || "—")}</td><td class="src-cell">${esc(s.source || "—")}</td>` +
      `<td class="reason-cell">${esc(s.reason || "—")}</td></tr>`;
  });
  document.getElementById("accuracy-provenance").innerHTML = h + "</tbody></table>";
  document.getElementById("accuracy-provenance-note").textContent =
    `Written by ${d.meta?.generated_by || "scripts/build_accuracy_json.py"} and archived ` +
    `daily under public/data/accuracy/, one dated file per day, never overwritten.`;
}

// ══════════════════════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════════════════════
async function init() {
  await loadData();
  initTooltip();
  initNav();
  renderPage("playoffs");
}

init();
