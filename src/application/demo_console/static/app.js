const state = {
  replay: null,
  selectedCase: null,
  cursor: 0,
  monitorInitialized: false,
  events: [],
  chartSamples: [],
  pendingDetections: [],
  liveMetrics: null,
  liveBaseline: null,
  replayMetrics: {inferences: 0, detections: 0, latencies: []}
};

const $ = (id) => document.getElementById(id);
const SVG_NS = "http://www.w3.org/2000/svg";
const CHART_SAMPLE_LIMIT = 80;

function emptyChartSample() {
  return {attack_count: 0, level: 0, predicted_class: "Benign", is_alert: false};
}

function resetChartSamples() {
  state.chartSamples = Array.from({length: CHART_SAMPLE_LIMIT}, emptyChartSample);
  state.pendingDetections = [];
}

function metricCounter(metrics, key) {
  return Number(metrics?.[key] || 0);
}

function percentile95(values) {
  if (!values.length) return null;
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.round((ordered.length - 1) * 0.95)];
}

function renderMetrics() {
  const live = state.liveMetrics || {};
  const baseline = state.liveBaseline || {};
  const liveWindows = Math.max(0, metricCounter(live, "windows") - metricCounter(baseline, "windows"));
  const liveAlerts = Math.max(0, metricCounter(live, "events") - metricCounter(baseline, "events"));
  const liveObservations = Math.max(0, metricCounter(live, "observations") - metricCounter(baseline, "observations"));
  const liveDrops = Math.max(0, metricCounter(live, "dropped_flows") - metricCounter(baseline, "dropped_flows"));
  const replayLatency = percentile95(state.replayMetrics.latencies);
  const liveLatency = live.inference_latency_ms_p95 == null ? null : Number(live.inference_latency_ms_p95);
  const displayedLatency = replayLatency ?? liveLatency;

  $("metric-windows").textContent = liveWindows + state.replayMetrics.inferences;
  $("metric-windows-source").textContent = `live ${liveWindows} · replay ${state.replayMetrics.inferences}`;
  $("metric-alerts").textContent = liveAlerts + state.replayMetrics.detections;
  $("metric-alerts-source").textContent = `live ${liveAlerts} · replay ${state.replayMetrics.detections}`;
  $("metric-latency").textContent = displayedLatency == null ? "—" : `${displayedLatency.toFixed(1)} ms`;
  $("metric-latency-source").textContent = replayLatency == null ? "live inference p95" : "replay round-trip p95";
  $("metric-drop").textContent = `${(liveDrops / Math.max(1, liveObservations) * 100).toFixed(2)}%`;
  $("metric-drop-source").textContent = `live ${liveDrops}/${liveObservations} flows`;
}

async function json(url, options = {}) {
  const response = await fetch(url, {headers: {"Content-Type": "application/json"}, ...options});
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
  return body;
}

function selectReplayCase(item) {
  state.selectedCase = item;
  document.querySelectorAll(".class-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.id === item.id);
  });
  const detail = $("class-detail");
  detail.className = "class-detail";
  detail.replaceChildren();

  const name = document.createElement("strong");
  name.textContent = item.expected_class;
  const help = document.createElement("p");
  help.textContent = item.profile.behavior;
  const indicators = document.createElement("ul");
  indicators.className = "indicator-list";
  item.profile.indicators.forEach((value) => {
    const indicator = document.createElement("li");
    indicator.textContent = value;
    indicators.append(indicator);
  });
  const meta = document.createElement("div");
  meta.className = "class-meta";
  const sample = item.sample_characteristics;
  [
    ["Sensor / head", `${item.sensor_id} / ${item.client_id}`],
    ["Flow / protocol", `${sample.flow_count} / ${sample.protocols.join(", ")}`],
    ["Service / state", `${sample.services.join(", ")} / ${sample.connection_states.join(", ")}`],
    ["Cổng đích", sample.destination_ports.join(", ")],
    ["Gói orig/resp", `${sample.total_origin_packets} / ${sample.total_response_packets}`],
    ["Byte orig/resp", `${sample.total_origin_bytes} / ${sample.total_response_bytes}`],
    ["Duration", sample.duration_seconds ? `${sample.duration_seconds.min.toFixed(3)}–${sample.duration_seconds.max.toFixed(3)}s` : "missing"]
  ].forEach(([label, value]) => {
    const cell = document.createElement("span");
    cell.textContent = label;
    const content = document.createElement("b");
    content.textContent = value;
    cell.append(content);
    meta.append(cell);
  });
  const limitation = document.createElement("small");
  limitation.className = "profile-limitation";
  limitation.textContent = item.profile.limitation;
  detail.append(name, help, indicators, meta, limitation);
  $("run-replay-button").disabled = false;
}

function renderReplayCatalog(catalog) {
  state.replay = catalog;
  $("replay-disclaimer").textContent = "Class selector dùng mẫu validation cố định; expected label không được gửi vào production inference request.";
  catalog.cases.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "class-button";
    button.dataset.id = item.id;
    button.textContent = item.expected_class;
    button.title = `${item.sensor_id} → head ${item.client_id} · ${item.window_flows} flows`;
    button.addEventListener("click", () => selectReplayCase(item));
    $("replay-cases").append(button);
  });
  if (catalog.cases.length) selectReplayCase(catalog.cases[0]);
}

function confidenceBucket(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function replayChartEvent(result, latencyMs) {
  return {
    predicted_class: result.predicted_class,
    trusted_predicted_class: result.predicted_class,
    client_id: result.client_id,
    severity: result.is_alert ? "medium" : "none",
    is_alert: result.is_alert,
    decision_status: result.decision_status,
    alert_threshold: result.alert_threshold,
    confidence_bucket: confidenceBucket(result.confidence),
    inference_latency_ms: latencyMs,
    source_type: "validation-replay",
    alert_decision_source: "validation-replay",
    head_disagreement_count: 0,
    head_predictions: {}
  };
}

async function runScientificReplay() {
  const item = state.selectedCase;
  if (!item) return;
  const buttons = [...document.querySelectorAll(".class-button")];
  buttons.forEach((button) => { button.disabled = true; });
  $("run-replay-button").disabled = true;
  $("replay-result").className = "replay-result";
  $("replay-result").textContent = `Running ${item.expected_class}…`;
  try {
    const started = performance.now();
    const result = await json(`/api/scientific-replay/${item.id}`, {method: "POST"});
    const latencyMs = performance.now() - started;
    state.replayMetrics.inferences += 1;
    state.replayMetrics.detections += result.is_alert ? 1 : 0;
    state.replayMetrics.latencies = [...state.replayMetrics.latencies, latencyMs].slice(-200);
    renderMetrics();
    const panel = $("replay-result");
    panel.replaceChildren();
    panel.classList.add(result.correct ? "correct" : "incorrect");
    const title = document.createElement("strong");
    title.textContent = `${result.expected_class} → ${result.predicted_class}`;
    const detail = document.createElement("small");
    detail.textContent = `confidence ${confidenceBucket(result.confidence)} · ${result.sensor_id} → head ${result.client_id} · ${result.window_flows} flows`;
    const decision = document.createElement("small");
    decision.className = `decision ${result.is_alert ? "accepted" : "rejected"}`;
    if (result.decision_status === "below-threshold") {
      decision.textContent = `Không phát cảnh báo: ${confidenceBucket(result.confidence)} < ngưỡng ${result.predicted_class} ${confidenceBucket(result.alert_threshold)}.`;
    } else if (result.decision_status === "alert") {
      decision.textContent = `Đủ ngưỡng cảnh báo ${result.predicted_class} (${confidenceBucket(result.alert_threshold)}).`;
    } else {
      decision.textContent = "Không phát cảnh báo: dự đoán thô là Benign.";
    }
    const top = document.createElement("small");
    top.textContent = `Top 3: ${result.top3.map((entry) => `${entry.class} ${(entry.probability * 100).toFixed(1)}%`).join(" · ")}`;
    panel.append(title, detail, decision, top);

    const event = replayChartEvent(result, latencyMs);
    state.events = [event, ...state.events].slice(0, 8);
    if (event.is_alert) state.pendingDetections.push(event);
    renderLatest(event);
    renderEventList();
    updateAlertBanner(event);
  } catch (error) {
    $("replay-result").className = "replay-result incorrect";
    $("replay-result").textContent = `Replay failed: ${error.message}`;
  } finally {
    buttons.forEach((button) => { button.disabled = false; });
    $("run-replay-button").disabled = false;
  }
}

function frequencyLevel(count) {
  if (count === 0) return 0;
  if (count <= 2) return 1;
  if (count <= 5) return 2;
  return 3;
}

function sampleAttackSignal() {
  const detections = state.pendingDetections.splice(0);
  const counts = new Map();
  detections.forEach((event) => {
    counts.set(event.predicted_class, (counts.get(event.predicted_class) || 0) + 1);
  });
  const dominant = [...counts.entries()].sort((left, right) => right[1] - left[1])[0];
  const sample = dominant ? {
    attack_count: detections.length,
    level: frequencyLevel(detections.length),
    predicted_class: dominant[0],
    is_alert: true,
    confidence_bucket: detections[detections.length - 1].confidence_bucket,
    source_type: detections.some((event) => event.source_type === "live") ? "live" : "validation-replay"
  } : emptyChartSample();
  state.chartSamples = [...state.chartSamples, sample].slice(-CHART_SAMPLE_LIMIT);
  renderChart();
  updateAlertBanner(sample);
}

function updateAlertBanner(event) {
  const banner = $("alert-banner");
  if (!event || event.predicted_class === "Benign" || event.decision_status === "below-threshold") {
    banner.className = "alert-banner normal";
    banner.querySelector(".alert-icon").textContent = "✓";
    $("alert-title").textContent = "Benign";
    if (event?.decision_status === "below-threshold") {
      $("alert-detail").textContent = `Dự đoán thô ${event.predicted_class} ${event.confidence_bucket}, dưới ngưỡng cảnh báo.`;
    } else {
      $("alert-detail").textContent = event ? "Không có detection đạt ngưỡng trong giây hiện tại." : "Đang chờ dự đoán mới.";
    }
    return;
  }
  banner.className = `alert-banner ${event.is_alert ? "danger" : "detection"}`;
  banner.querySelector(".alert-icon").textContent = "!";
  $("alert-title").textContent = event.predicted_class;
  if (event.source_type === "validation-replay") {
    $("alert-detail").textContent = `Validation replay đạt ngưỡng; ${event.attack_count || 1} detection/s, không phải live policy alert.`;
  } else if (event.is_alert) {
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
    x: 24 + (index / Math.max(1, CHART_SAMPLE_LIMIT - 1)) * 956,
    y: 180 - sample.level * 48,
    sample
  }));
  $("detection-line").setAttribute("points", coordinates.map((point) => `${point.x.toFixed(1)},${point.y}`).join(" "));
  const hasDetection = samples.some((sample) => sample.attack_count > 0);
  $("detection-line").setAttribute("class", `detection-line ${hasDetection ? "danger" : "normal"}`);
  const last = coordinates[coordinates.length - 1];
  $("detection-area").setAttribute("d", `M ${coordinates.map((point) => `${point.x.toFixed(1)} ${point.y}`).join(" L ")} L ${last.x.toFixed(1)} 180 L 24 180 Z`);
  $("detection-area").classList.toggle("active", hasDetection);
  $("detection-points").replaceChildren();

  const attackPoints = coordinates.filter((point) => point.sample.attack_count > 0);
  attackPoints.forEach((point, index) => {
    const group = document.createElementNS(SVG_NS, "g");
    group.setAttribute("class", "chart-point danger");
    const circle = document.createElementNS(SVG_NS, "circle");
    circle.setAttribute("cx", point.x.toFixed(1));
    circle.setAttribute("cy", point.y);
    circle.setAttribute("r", "6");
    const tooltip = document.createElementNS(SVG_NS, "title");
    tooltip.textContent = `${point.sample.predicted_class} · ${point.sample.attack_count} detection/s`;
    circle.append(tooltip);
    group.append(circle);
    if (index >= attackPoints.length - 3) {
      const label = document.createElementNS(SVG_NS, "text");
      label.setAttribute("x", Math.min(910, Math.max(45, point.x)).toFixed(1));
      label.setAttribute("y", Math.max(20, point.y - 12));
      label.textContent = `${point.sample.predicted_class} · ${point.sample.attack_count}/s`;
      group.append(label);
    }
    $("detection-points").append(group);
  });
  const current = samples[samples.length - 1];
  $("chart-window-count").textContent = `80s · ${current.attack_count} detection/s`;
}

function renderLatest(event) {
  if (!event) {
    $("latest-class").textContent = "Waiting…";
    $("latest-detail").textContent = "Chưa có flow mới.";
    return;
  }
  $("latest-class").textContent = event.predicted_class;
  const source = event.source_type === "validation-replay" ? "validation replay" : "live collector";
  $("latest-detail").textContent = `${source} · ${event.client_id || "—"} · confidence ${event.confidence_bucket}`;
}

function eventNode(event) {
  const row = document.createElement("div");
  row.className = `event${event.is_alert ? " alert" : (event.predicted_class !== "Benign" ? " detection" : "")}`;
  const mark = document.createElement("i"); mark.className = "event-mark";
  const main = document.createElement("div"); main.className = "event-main";
  const label = document.createElement("strong"); label.textContent = event.predicted_class;
  const detail = document.createElement("small");
  if (event.source_type === "validation-replay") {
    detail.textContent = event.decision_status === "below-threshold" ? "validation replay · dưới ngưỡng" : "validation replay · đạt ngưỡng";
  } else {
    detail.textContent = `${event.alert_decision_source || "live"} · trusted ${event.trusted_predicted_class || event.predicted_class}`;
  }
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
        state.pendingDetections.push(...fresh.filter((event) => event.is_alert));
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
  state.replayMetrics = {inferences: 0, detections: 0, latencies: []};
  resetChartSamples();
  renderLatest(null);
  renderEventList();
  renderMetrics();
  renderChart();
}

async function boot() {
  resetChartSamples();
  renderChart();
  try {
    renderReplayCatalog(await json("/api/scientific-replay"));
    $("system-dot").className = "status-dot ready";
    $("system-label").textContent = "Console ready";
  } catch (error) {
    $("system-dot").className = "status-dot error";
    $("system-label").textContent = error.message;
  }
  $("run-replay-button").addEventListener("click", runScientificReplay);
  $("clear-events").addEventListener("click", clearMonitor);
  setInterval(pollMonitor, 1000);
  setInterval(sampleAttackSignal, 1000);
  await pollMonitor();
}

boot();
