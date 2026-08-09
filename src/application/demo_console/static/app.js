const state = {
  replay: null,
  selectedCase: null,
  cursor: 0,
  events: [],
  chartEvents: []
};

const $ = (id) => document.getElementById(id);
const SVG_NS = "http://www.w3.org/2000/svg";
const severityLevel = {none: 0, low: 1, medium: 2, high: 3};
const classHelp = {
  "Benign": "Lưu lượng bình thường theo nhãn huấn luyện.",
  "Attack": "Nhãn tấn công tổng quát, chủ yếu gồm hành vi SSH trong dữ liệu nguồn.",
  "C&C": "Lưu lượng command-and-control đã biết trong IoT-23.",
  "C&C-HeartBeat": "Nhịp liên lạc định kỳ giữa thiết bị và máy chủ C&C.",
  "DDoS": "Lưu lượng distributed denial-of-service trong dữ liệu huấn luyện.",
  "Okiru": "Hành vi thuộc họ malware Okiru trong dữ liệu huấn luyện.",
  "PartOfAHorizontalPortScan": "Quét cùng một cổng trên nhiều máy đích."
};

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
  help.textContent = classHelp[item.expected_class] || "Model output class.";
  const meta = document.createElement("div");
  meta.className = "class-meta";
  [
    ["Sensor", item.sensor_id],
    ["Trusted head", item.client_id],
    ["Window", `${item.window_flows} flows`]
  ].forEach(([label, value]) => {
    const cell = document.createElement("span");
    cell.textContent = label;
    const content = document.createElement("b");
    content.textContent = value;
    cell.append(content);
    meta.append(cell);
  });
  detail.append(name, help, meta);
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

function replayChartEvent(result) {
  return {
    predicted_class: result.predicted_class,
    trusted_predicted_class: result.predicted_class,
    client_id: result.client_id,
    severity: result.predicted_class === "Benign" ? "none" : "medium",
    is_alert: false,
    confidence_bucket: confidenceBucket(result.confidence),
    inference_latency_ms: "replay",
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
    const result = await json(`/api/scientific-replay/${item.id}`, {method: "POST"});
    const panel = $("replay-result");
    panel.replaceChildren();
    panel.classList.add(result.correct ? "correct" : "incorrect");
    const title = document.createElement("strong");
    title.textContent = `${result.expected_class} → ${result.predicted_class}`;
    const detail = document.createElement("small");
    detail.textContent = `confidence ${confidenceBucket(result.confidence)} · ${result.sensor_id} → head ${result.client_id} · ${result.window_flows} flows`;
    const top = document.createElement("small");
    top.textContent = `Top 3: ${result.top3.map((entry) => `${entry.class} ${(entry.probability * 100).toFixed(1)}%`).join(" · ")}`;
    panel.append(title, detail, top);

    const event = replayChartEvent(result);
    state.events = [event, ...state.events].slice(0, 4);
    state.chartEvents = [...state.chartEvents, event].slice(-80);
    renderLatest(event);
    renderEventList();
    renderChart();
  } catch (error) {
    $("replay-result").className = "replay-result incorrect";
    $("replay-result").textContent = `Replay failed: ${error.message}`;
  } finally {
    buttons.forEach((button) => { button.disabled = false; });
    $("run-replay-button").disabled = false;
  }
}

function eventLevel(event) {
  if (event.predicted_class === "Benign") return 0;
  return severityLevel[event.severity] || 1;
}

function updateAlertBanner(event) {
  const banner = $("alert-banner");
  if (!event || event.predicted_class === "Benign") {
    banner.className = "alert-banner normal";
    banner.querySelector(".alert-icon").textContent = "✓";
    $("alert-title").textContent = "Benign";
    $("alert-detail").textContent = event ? "Model prediction is Benign." : "Đang chờ dự đoán mới.";
    return;
  }
  banner.className = `alert-banner ${event.is_alert ? "danger" : "detection"}`;
  banner.querySelector(".alert-icon").textContent = "!";
  $("alert-title").textContent = event.predicted_class;
  if (event.source_type === "validation-replay") {
    $("alert-detail").textContent = "Validation replay detection; không phải live policy alert.";
  } else if (event.is_alert) {
    $("alert-detail").textContent = "Live prediction reached the policy alert gate.";
  } else if (event.alert_decision_source === "trusted-shadow") {
    $("alert-detail").textContent = `6-head fusion detection · shadow mode · trusted head: ${event.trusted_predicted_class}.`;
  } else {
    $("alert-detail").textContent = "Model detection is below the policy alert gate.";
  }
}

function renderChart() {
  const samples = state.chartEvents.slice(-80);
  const coordinates = [{x: 24, y: 180, event: null}];
  samples.forEach((event, index) => {
    const x = 24 + ((index + 1) / Math.max(1, samples.length)) * 956;
    coordinates.push({x, y: 180 - eventLevel(event) * 48, event});
  });
  if (!samples.length) coordinates.push({x: 980, y: 180, event: null});
  $("detection-line").setAttribute("points", coordinates.map((point) => `${point.x.toFixed(1)},${point.y}`).join(" "));
  const hasPolicyAlert = samples.some((event) => event.is_alert);
  const hasDetection = samples.some((event) => event.predicted_class !== "Benign");
  $("detection-line").setAttribute("class", `detection-line ${hasPolicyAlert ? "danger" : (hasDetection ? "detection" : "normal")}`);
  const last = coordinates[coordinates.length - 1];
  $("detection-area").setAttribute("d", `M ${coordinates.map((point) => `${point.x.toFixed(1)} ${point.y}`).join(" L ")} L ${last.x.toFixed(1)} 180 L 24 180 Z`);
  $("detection-area").classList.toggle("active", hasDetection);
  $("detection-points").replaceChildren();

  const attackPoints = coordinates.filter((point) => point.event && point.event.predicted_class !== "Benign");
  attackPoints.forEach((point, index) => {
    const group = document.createElementNS(SVG_NS, "g");
    group.setAttribute("class", point.event.is_alert ? "chart-point danger" : "chart-point detection");
    const circle = document.createElementNS(SVG_NS, "circle");
    circle.setAttribute("cx", point.x.toFixed(1));
    circle.setAttribute("cy", point.y);
    circle.setAttribute("r", "6");
    const tooltip = document.createElementNS(SVG_NS, "title");
    tooltip.textContent = `${point.event.predicted_class} · ${point.event.confidence_bucket}`;
    circle.append(tooltip);
    group.append(circle);
    if (index >= attackPoints.length - 3) {
      const label = document.createElementNS(SVG_NS, "text");
      label.setAttribute("x", Math.min(910, Math.max(45, point.x)).toFixed(1));
      label.setAttribute("y", Math.max(20, point.y - 12));
      label.textContent = point.event.predicted_class;
      group.append(label);
    }
    $("detection-points").append(group);
  });
  $("chart-window-count").textContent = `${samples.length} points`;
  updateAlertBanner(samples[samples.length - 1]);
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
  detail.textContent = event.source_type === "validation-replay" ? "validation replay" : `${event.alert_decision_source || "live"} · trusted ${event.trusted_predicted_class || event.predicted_class}`;
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
  $("event-list").replaceChildren(...state.events.slice(0, 3).map(eventNode));
}

function renderHeadDiagnostics(event) {
  const entries = Object.entries(event?.head_predictions || {});
  if (!entries.length) {
    $("head-disagreement").textContent = event?.source_type === "validation-replay" ? "Replay result" : "No data";
    $("head-grid").innerHTML = '<div class="empty-event">Head diagnostics xuất hiện với live collector event.</div>';
    return;
  }
  $("head-disagreement").textContent = `${event.head_disagreement_count}/6 disagree`;
  $("head-grid").replaceChildren(...entries.map(([head, prediction]) => {
    const card = document.createElement("article");
    const isTrusted = head === event.client_id;
    const agrees = prediction.predicted_label === event.predicted_class;
    card.className = `head-card${isTrusted ? " trusted" : ""}${agrees ? " agrees" : " disagrees"}`;
    const name = document.createElement("small"); name.textContent = `HEAD ${head}${isTrusted ? " · TRUSTED" : ""}`;
    const label = document.createElement("strong"); label.textContent = prediction.predicted_label;
    const confidence = document.createElement("span"); confidence.textContent = prediction.confidence_bucket;
    card.append(name, label, confidence);
    return card;
  }));
}

async function pollMonitor() {
  try {
    const body = await json(`/api/monitor?after=${state.cursor}&limit=100`);
    if (body.events.length) {
      state.cursor = body.next_cursor;
      const fresh = body.events.map((event) => ({...event, source_type: "live"}));
      state.chartEvents = [...state.chartEvents, ...fresh].slice(-80);
      state.events = [...fresh.slice().reverse(), ...state.events].slice(0, 4);
      const latest = fresh[fresh.length - 1];
      renderLatest(latest);
      renderEventList();
      renderHeadDiagnostics(latest);
      renderChart();
    }
    const metrics = body.metrics;
    $("metric-windows").textContent = metrics.windows;
    $("metric-alerts").textContent = metrics.events;
    $("metric-latency").textContent = metrics.inference_latency_ms_p95 == null ? "—" : `${Number(metrics.inference_latency_ms_p95).toFixed(1)} ms`;
    $("metric-drop").textContent = `${(Number(metrics.dropped_flows) / Math.max(1, Number(metrics.observations)) * 100).toFixed(2)}%`;
    $("system-dot").className = "status-dot ready";
    $("system-label").textContent = "Pipeline ready";
  } catch (_) {
    $("system-dot").className = "status-dot error";
    $("system-label").textContent = "Monitor offline";
  }
}

function clearMonitor() {
  state.events = [];
  state.chartEvents = [];
  renderLatest(null);
  renderEventList();
  renderHeadDiagnostics(null);
  renderChart();
}

async function boot() {
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
  await pollMonitor();
}

boot();
