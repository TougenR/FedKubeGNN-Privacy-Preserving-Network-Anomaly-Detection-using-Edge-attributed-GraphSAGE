const state = {
  cursor: 0,
  monitorInitialized: false,
  events: [],
  chartSamples: [],
  pendingSignals: [],
  liveMetrics: null,
  liveBaseline: null
};

const $ = (id) => document.getElementById(id);
const SVG_NS = "http://www.w3.org/2000/svg";
const CHART_SAMPLE_LIMIT = 80;

function emptyChartSample() {
  return {attack_count: 0, level: 0, predicted_class: "Benign", is_alert: false};
}

function resetChartSamples() {
  state.chartSamples = Array.from({length: CHART_SAMPLE_LIMIT}, emptyChartSample);
  state.pendingSignals = [];
}

function metricCounter(metrics, key) {
  return Number(metrics?.[key] || 0);
}

function renderMetrics() {
  const live = state.liveMetrics || {};
  const baseline = state.liveBaseline || {};
  const liveWindows = Math.max(0, metricCounter(live, "windows") - metricCounter(baseline, "windows"));
  const liveAlerts = Math.max(0, metricCounter(live, "events") - metricCounter(baseline, "events"));
  const liveObservations = Math.max(0, metricCounter(live, "observations") - metricCounter(baseline, "observations"));
  const liveDrops = Math.max(0, metricCounter(live, "dropped_flows") - metricCounter(baseline, "dropped_flows"));
  const liveLatency = live.inference_latency_ms_p95 == null ? null : Number(live.inference_latency_ms_p95);

  $("metric-windows").textContent = liveWindows;
  $("metric-windows-source").textContent = `live ${liveWindows}`;
  $("metric-alerts").textContent = liveAlerts;
  $("metric-alerts-source").textContent = `live ${liveAlerts}`;
  $("metric-latency").textContent = liveLatency == null ? "—" : `${liveLatency.toFixed(1)} ms`;
  $("metric-latency-source").textContent = "live inference p95";
  $("metric-drop").textContent = `${(liveDrops / Math.max(1, liveObservations) * 100).toFixed(2)}%`;
  $("metric-drop-source").textContent = `live ${liveDrops}/${liveObservations} flows`;
}

async function json(url, options = {}) {
  const response = await fetch(url, {headers: {"Content-Type": "application/json"}, ...options});
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
  return body;
}

function frequencyLevel(count) {
  if (count === 0) return 0;
  if (count <= 2) return 1;
  if (count <= 5) return 2;
  return 3;
}

function smoothSegment(previous, current) {
  const midpoint = (previous.x + current.x) / 2;
  return `C ${midpoint.toFixed(1)} ${previous.y} ${midpoint.toFixed(1)} ${current.y} ${current.x.toFixed(1)} ${current.y}`;
}

function smoothChartPath(coordinates) {
  if (!coordinates.length) return "";
  return coordinates.slice(1).reduce(
    (path, current, index) => `${path} ${smoothSegment(coordinates[index], current)}`,
    `M ${coordinates[0].x.toFixed(1)} ${coordinates[0].y}`
  );
}

function sampleAttackSignal() {
  const detections = state.pendingSignals.splice(0);
  const counts = new Map();
  detections.forEach((event) => {
    counts.set(event.predicted_class, (counts.get(event.predicted_class) || 0) + 1);
  });
  const dominant = [...counts.entries()].sort((left, right) => right[1] - left[1])[0];
  const dominantEvent = dominant
    ? [...detections].reverse().find((event) => event.predicted_class === dominant[0])
    : null;
  const alertCount = detections.filter((event) => event.is_alert).length;
  const sample = dominant ? {
    attack_count: detections.length,
    alert_count: alertCount,
    level: frequencyLevel(detections.length),
    predicted_class: dominant[0],
    is_alert: alertCount > 0,
    confidence_bucket: dominantEvent.confidence_bucket,
    source_type: "live",
    alert_decision_source: dominantEvent.alert_decision_source,
    trusted_predicted_class: dominantEvent.trusted_predicted_class,
    decision_status: dominantEvent.decision_status
  } : emptyChartSample();
  state.chartSamples = [...state.chartSamples, sample].slice(-CHART_SAMPLE_LIMIT);
  renderChart();
  updateAlertBanner(sample);
}

function updateAlertBanner(event) {
  const banner = $("alert-banner");
  if (!event || event.predicted_class === "Benign") {
    banner.className = "alert-banner normal";
    banner.querySelector(".alert-icon").textContent = "✓";
    $("alert-title").textContent = "Benign";
    $("alert-detail").textContent = event ? "Không có model detection trong giây hiện tại." : "Đang chờ dự đoán mới.";
    return;
  }
  banner.className = `alert-banner ${event.is_alert ? "danger" : "detection"}`;
  banner.querySelector(".alert-icon").textContent = "!";
  $("alert-title").textContent = event.predicted_class;
  if (event.is_alert) {
    $("alert-detail").textContent = `${event.attack_count || 1} detection/s đã đạt ngưỡng policy.`;
  } else if (event.alert_decision_source === "trusted-shadow") {
    $("alert-detail").textContent = `6-head fusion detection · shadow mode · trusted head: ${event.trusted_predicted_class}.`;
  } else {
    $("alert-detail").textContent = "Model detection is below the policy alert gate.";
  }
}

function renderChart() {
  const samples = state.chartSamples.slice(-CHART_SAMPLE_LIMIT);
  const coordinates = samples.map((sample, index) => ({
    index,
    x: 24 + (index / Math.max(1, CHART_SAMPLE_LIMIT - 1)) * 956,
    y: 180 - sample.level * 48,
    sample
  }));
  const smoothPath = smoothChartPath(coordinates);
  $("detection-line").setAttribute("d", smoothPath);
  const hasDetection = samples.some((sample) => sample.attack_count > 0);
  $("detection-line").setAttribute("class", "detection-line normal");
  const last = coordinates[coordinates.length - 1];
  $("detection-area").setAttribute("d", `${smoothPath} L ${last.x.toFixed(1)} 180 L 24 180 Z`);
  $("detection-area").setAttribute("class", `detection-area ${hasDetection ? "signal" : "normal"}`);
  $("detection-segments").replaceChildren();
  coordinates.slice(1).forEach((point, index) => {
    const previous = coordinates[index];
    const signal = point.sample.attack_count > 0 ? point.sample : previous.sample;
    if (signal.attack_count === 0) return;
    const segment = document.createElementNS(SVG_NS, "path");
    segment.setAttribute("d", `M ${previous.x.toFixed(1)} ${previous.y} ${smoothSegment(previous, point)}`);
    segment.setAttribute("class", `chart-segment ${signal.is_alert ? "danger" : "detection"}`);
    $("detection-segments").append(segment);
  });
  $("detection-points").replaceChildren();

  const attackPoints = coordinates.filter((point) => point.sample.attack_count > 0);
  const labelPoints = new Set(
    attackPoints.filter((point) => {
      const next = coordinates[point.index + 1];
      return !next
        || next.sample.attack_count === 0
        || next.sample.predicted_class !== point.sample.predicted_class
        || next.sample.is_alert !== point.sample.is_alert;
    }).slice(-3)
  );
  attackPoints.forEach((point) => {
    const group = document.createElementNS(SVG_NS, "g");
    group.setAttribute("class", `chart-point ${point.sample.is_alert ? "danger" : "detection"}`);
    const circle = document.createElementNS(SVG_NS, "circle");
    circle.setAttribute("cx", point.x.toFixed(1));
    circle.setAttribute("cy", point.y);
    circle.setAttribute("r", point.sample.is_alert ? "5.5" : "5");
    const tooltip = document.createElementNS(SVG_NS, "title");
    const signalKind = point.sample.is_alert ? "policy alert" : "model detection";
    tooltip.textContent = `${point.sample.predicted_class} · ${point.sample.attack_count}/s · ${signalKind}`;
    circle.append(tooltip);
    group.append(circle);
    if (labelPoints.has(point)) {
      const label = document.createElementNS(SVG_NS, "text");
      label.setAttribute("x", Math.min(910, Math.max(45, point.x)).toFixed(1));
      label.setAttribute("y", Math.max(20, point.y - 12));
      label.textContent = `${point.sample.predicted_class} · ${point.sample.attack_count}/s`;
      group.append(label);
    }
    $("detection-points").append(group);
  });
  const current = samples[samples.length - 1];
  $("chart-window-count").textContent = `80s · ${current.attack_count} detection/s · ${current.alert_count || 0} alert`;
}

function renderLatest(event) {
  if (!event) {
    $("latest-class").textContent = "Waiting…";
    $("latest-detail").textContent = "Chưa có flow mới.";
    return;
  }
  $("latest-class").textContent = event.predicted_class;
  const run = ` · run ${event.run_id || "untracked"}`;
  $("latest-detail").textContent = `live collector${run} · ${event.client_id || "—"} · confidence ${event.confidence_bucket}`;
}

function eventNode(event) {
  const row = document.createElement("div");
  row.className = `event${event.is_alert ? " alert" : (event.predicted_class !== "Benign" ? " detection" : "")}`;
  const mark = document.createElement("i"); mark.className = "event-mark";
  const main = document.createElement("div"); main.className = "event-main";
  const label = document.createElement("strong"); label.textContent = event.predicted_class;
  const detail = document.createElement("small");
  detail.textContent = `${event.alert_decision_source || "live"} · ${event.run_id || "untracked"} · trusted ${event.trusted_predicted_class || event.predicted_class}`;
  main.append(label, detail);
  const meta = document.createElement("div"); meta.className = "event-meta"; meta.textContent = event.confidence_bucket;
  row.append(mark, main, meta);
  return row;
}

function renderEventList() {
  if (!state.events.length) {
    $("event-list").innerHTML = '<div class="empty-event">No predictions</div>';
    return;
  }
  $("event-list").replaceChildren(...state.events.slice(0, 6).map(eventNode));
}

async function pollMonitor() {
  try {
    const body = await json(`/api/monitor?after=${state.cursor}&limit=100`);
    const initialSnapshot = !state.monitorInitialized;
    if (body.events.length) {
      state.cursor = body.next_cursor;
      const fresh = body.events.map((event) => ({...event, source_type: "live"}));
      if (!initialSnapshot) {
        state.pendingSignals.push(...fresh.filter((event) => event.predicted_class !== "Benign"));
      }
      state.events = [...fresh.slice().reverse(), ...state.events].slice(0, 8);
      const latest = fresh[fresh.length - 1];
      renderLatest(latest);
      renderEventList();
      updateAlertBanner(latest);
    }
    if (initialSnapshot && body.events.length < 100) state.monitorInitialized = true;
    const metrics = body.metrics;
    if (state.liveBaseline == null) {
      state.liveBaseline = {...metrics};
    }
    state.liveMetrics = metrics;
    renderMetrics();
    $("system-dot").className = "status-dot ready";
    $("system-label").textContent = "Pipeline ready";
  } catch (_) {
    $("system-dot").className = "status-dot error";
    $("system-label").textContent = "Monitor offline";
  }
}

function clearMonitor() {
  state.events = [];
  state.liveBaseline = state.liveMetrics == null ? null : {...state.liveMetrics};
  resetChartSamples();
  renderLatest(null);
  renderEventList();
  renderMetrics();
  renderChart();
}

async function boot() {
  resetChartSamples();
  renderChart();
  $("clear-events").addEventListener("click", clearMonitor);
  setInterval(pollMonitor, 1000);
  setInterval(sampleAttackSignal, 1000);
  await pollMonitor();
}

boot();
