const state = {
  catalog: null,
  replay: null,
  selected: null,
  cursor: 0,
  events: [],
  chartEvents: []
};
const $ = (id) => document.getElementById(id);
const SVG_NS = "http://www.w3.org/2000/svg";
const classDisplay = {
  "Benign": "Bình thường",
  "Attack": "Tấn công chung",
  "C&C": "Điều khiển và chỉ huy (C&C)",
  "C&C-HeartBeat": "Nhịp tim C&C",
  "DDoS": "Từ chối dịch vụ phân tán (DDoS)",
  "Okiru": "Mã độc Okiru",
  "PartOfAHorizontalPortScan": "Quét cổng ngang"
};
const classHelp = {
  "Benign": "Flow được model xem là lưu lượng bình thường.",
  "Attack": "Nhãn tấn công tổng quát trong bộ dữ liệu IoT-23.",
  "C&C": "Hành vi điều khiển và chỉ huy đã biết trong IoT-23.",
  "C&C-HeartBeat": "Nhịp liên lạc định kỳ giữa thiết bị và máy chủ C&C.",
  "DDoS": "Flow thuộc lớp từ chối dịch vụ phân tán.",
  "Okiru": "Họ hành vi mã độc Okiru trong tập huấn luyện.",
  "PartOfAHorizontalPortScan": "Flow thuộc hoạt động quét cùng một cổng trên nhiều máy đích."
};
const parameterDisplay = {
  request_count: "Số yêu cầu",
  interval_ms: "Khoảng cách",
  bursts: "Số đợt",
  concurrency: "Mức đồng thời",
  pause_ms: "Thời gian nghỉ",
  requests_per_second: "Yêu cầu mỗi giây",
  duration_seconds: "Thời lượng",
  connections: "Số kết nối",
  delay_ms: "Độ trễ",
  port_count: "Số cổng",
  events: "Số sự kiện"
};
const runStatus = {
  running: "ĐANG CHẠY",
  completed: "HOÀN TẤT",
  cancelled: "ĐÃ DỪNG",
  failed: "THẤT BẠI"
};
const severityLevel = {none: 0, low: 1, medium: 2, high: 3};

function displayClass(name) {
  return classDisplay[name] || name;
}

async function json(url, options = {}) {
  const response = await fetch(url, {headers: {"Content-Type": "application/json"}, ...options});
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || `Lỗi HTTP ${response.status}`);
  return body;
}

function selectScenario(scenario) {
  state.selected = scenario;
  document.querySelectorAll(".scenario-card").forEach((card) => card.classList.toggle("active", card.dataset.id === scenario.id));
  $("category-pill").textContent = scenario.category;
  $("scenario-detail").classList.remove("empty-state");
  $("scenario-detail").replaceChildren();
  const title = document.createElement("h3"); title.textContent = scenario.display_name;
  const mechanism = document.createElement("p"); mechanism.textContent = scenario.mechanism;
  const limitation = document.createElement("p"); limitation.className = "limitation"; limitation.textContent = `Lưu ý: ${scenario.limitation}`;
  $("scenario-detail").append(title, mechanism, limitation);
  $("parameter-form").replaceChildren();
  Object.entries(scenario.bounds).forEach(([name, bound]) => {
    const row = document.createElement("div"); row.className = "field";
    const label = document.createElement("label"); label.textContent = parameterDisplay[name] || name.replaceAll("_", " ");
    const hint = document.createElement("small"); hint.textContent = `${bound.minimum}–${bound.maximum} ${bound.unit}`; label.append(hint);
    const input = document.createElement("input"); input.type = "number"; input.name = name; input.min = bound.minimum; input.max = bound.maximum; input.value = scenario.defaults[name]; input.required = true;
    row.append(label, input); $("parameter-form").append(row);
  });
  $("start-button").disabled = false;
}

function renderCatalog(catalog) {
  $("disclaimer").textContent = catalog.disclaimer;
  if (catalog.observation_mode === "zeek") {
    $("queued-label").textContent = "Hàng đợi adapter (không dùng)";
    $("delivered-label").textContent = "Adapter chuyển giao (không dùng)";
    $("queued").textContent = "—";
    $("delivered").textContent = "—";
  }
  catalog.scenarios.forEach((scenario) => {
    const card = document.createElement("button"); card.type = "button"; card.className = "scenario-card"; card.dataset.id = scenario.id;
    const title = document.createElement("strong"); title.textContent = scenario.display_name;
    const summary = document.createElement("span"); summary.textContent = scenario.summary;
    card.append(title, summary); card.addEventListener("click", () => selectScenario(scenario)); $("scenario-grid").append(card);
  });
  catalog.model_classes.forEach((name) => {
    const card = document.createElement("article"); card.className = "model-class";
    const title = document.createElement("strong"); title.textContent = displayClass(name);
    const contract = document.createElement("code"); contract.textContent = `Nhãn model: ${name}`;
    const help = document.createElement("small"); help.textContent = classHelp[name] || "Lớp đầu ra của model.";
    card.append(title, contract, help); $("class-list").append(card);
  });
}

function renderReplayCatalog(catalog) {
  state.replay = catalog;
  $("replay-disclaimer").textContent = catalog.disclaimer;
  catalog.cases.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button"; button.className = "replay-case";
    button.textContent = item.display_name;
    button.title = `${item.sensor_id} → đầu phân loại ${item.client_id} · ${item.window_flows} flow`;
    button.addEventListener("click", () => runScientificReplay(item, button));
    $("replay-cases").append(button);
  });
}

async function runScientificReplay(item, button) {
  const buttons = [...document.querySelectorAll(".replay-case")];
  buttons.forEach((candidate) => { candidate.disabled = true; });
  $("replay-result").className = "replay-result";
  $("replay-result").textContent = `Đang phát lại mẫu ${item.display_name}…`;
  try {
    const result = await json(`/api/scientific-replay/${item.id}`, {method: "POST"});
    $("replay-result").replaceChildren();
    $("replay-result").classList.add(result.correct ? "correct" : "incorrect");
    const title = document.createElement("strong");
    title.textContent = `${displayClass(result.expected_class)} → ${displayClass(result.predicted_class)}`;
    const detail = document.createElement("small");
    detail.textContent = `chỉ dùng validation · ${result.sensor_id} → đầu ${result.client_id} · độ tin cậy ${result.confidence.toFixed(4)} · ${result.window_flows} flow · gửi nhãn thật vào inference: ${result.request_contains_ground_truth ? "CÓ" : "KHÔNG"}`;
    const top = document.createElement("small");
    top.textContent = `Ba xác suất cao nhất: ${result.top3.map((entry) => `${displayClass(entry.class)} ${(entry.probability * 100).toFixed(2)}%`).join(" · ")}`;
    $("replay-result").append(title, detail, top);
  } catch (error) {
    $("replay-result").className = "replay-result incorrect";
    $("replay-result").textContent = `Phát lại thất bại: ${error.message}`;
  } finally {
    buttons.forEach((candidate) => { candidate.disabled = false; });
    button.focus();
  }
}

function runParameters() {
  return Object.fromEntries([...new FormData($("parameter-form")).entries()].map(([key, value]) => [key, Number(value)]));
}

async function startRun(event) {
  event.preventDefault();
  if (!state.selected) return;
  try {
    await json("/api/runs", {method: "POST", body: JSON.stringify({scenario_id: state.selected.id, parameters: runParameters()})});
    $("start-button").disabled = true; $("stop-button").disabled = false;
  } catch (error) { alert(`Không thể chạy: ${error.message}`); }
}

async function stopRun() {
  try { await json("/api/runs/current", {method: "DELETE"}); } catch (error) { alert(error.message); }
}

function expectedAttempts(run) {
  const p = run.parameters;
  if (run.scenario_id === "benign-browsing") return p.request_count;
  if (run.scenario_id === "connection-burst") return p.bursts * p.concurrency;
  if (run.scenario_id === "request-flood") return p.requests_per_second * p.duration_seconds;
  if (run.scenario_id === "slow-connections") return p.connections;
  if (run.scenario_id === "port-probe") return p.port_count;
  return p.events;
}

async function pollRun() {
  try {
    const {run} = await json("/api/runs/current");
    if (!run) return;
    $("run-status").textContent = runStatus[run.status] || run.status.toUpperCase(); $("run-id").textContent = run.run_id;
    $("attempted").textContent = run.attempted; $("succeeded").textContent = run.succeeded; $("failed").textContent = run.failed;
    const delivery = run.pipeline?.delivery || {};
    const collector = run.pipeline?.collector || {};
    $("queued").textContent = delivery.available === false && state.catalog?.observation_mode === "zeek" ? "—" : (delivery.enqueued || 0);
    $("delivered").textContent = delivery.available === false && state.catalog?.observation_mode === "zeek" ? "—" : (delivery.delivered || 0);
    $("collected").textContent = collector.accepted || 0;
    $("predicted").textContent = collector.predicted || 0;
    $("run-late-dropped").textContent = collector.late_dropped || 0;
    $("delivery-failed").textContent = (delivery.terminal_failures || 0) + (delivery.queue_dropped || 0);
    $("progress-bar").style.width = `${Math.min(100, run.attempted / Math.max(1, expectedAttempts(run)) * 100)}%`;
    const active = run.status === "running"; $("start-button").disabled = active || !state.selected; $("stop-button").disabled = !active;
  } catch (_) { /* Đèn trạng thái hệ thống thể hiện lỗi kết nối. */ }
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
    $("alert-title").textContent = "Bình thường";
    $("alert-detail").textContent = event ? "Dự đoán mới nhất là lưu lượng bình thường." : "Chưa ghi nhận dự đoán tấn công.";
    return;
  }
  banner.className = `alert-banner ${event.is_alert ? "danger" : "detection"}`;
  banner.querySelector(".alert-icon").textContent = "!";
  $("alert-title").textContent = `${event.is_alert ? "Cảnh báo" : "Phát hiện"}: ${displayClass(event.predicted_class)}`;
  if (event.is_alert) {
    $("alert-detail").textContent = `Model phát hiện ${event.predicted_class} và kết quả đã đạt ngưỡng cảnh báo chính sách.`;
  } else if (event.alert_decision_source === "trusted-shadow") {
    $("alert-detail").textContent = `Fusion 6 head phát hiện ${event.predicted_class}; đang chạy shadow nên cảnh báo vẫn theo trusted head (${displayClass(event.trusted_predicted_class)}).`;
  } else {
    $("alert-detail").textContent = `Model phát hiện ${event.predicted_class}, nhưng độ tin cậy chưa đạt ngưỡng cảnh báo chính sách.`;
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
  const points = coordinates.map((point) => `${point.x.toFixed(1)},${point.y}`).join(" ");
  $("detection-line").setAttribute("points", points);
  const hasPolicyAlert = samples.some((event) => event.is_alert);
  const hasDetection = samples.some((event) => event.predicted_class !== "Benign");
  $("detection-line").setAttribute("class", `detection-line ${hasPolicyAlert ? "danger" : (hasDetection ? "detection" : "normal")}`);
  const last = coordinates[coordinates.length - 1];
  const areaPath = `M ${coordinates.map((point) => `${point.x.toFixed(1)} ${point.y}`).join(" L ")} L ${last.x.toFixed(1)} 180 L 24 180 Z`;
  $("detection-area").setAttribute("d", areaPath);
  $("detection-area").classList.toggle("active", hasDetection);
  $("detection-points").replaceChildren();
  const attackPoints = coordinates.filter((point) => point.event && point.event.predicted_class !== "Benign");
  attackPoints.forEach((point, index) => {
    const group = document.createElementNS(SVG_NS, "g");
    group.setAttribute("class", point.event.is_alert ? "chart-point danger" : "chart-point detection");
    const circle = document.createElementNS(SVG_NS, "circle");
    circle.setAttribute("cx", point.x.toFixed(1)); circle.setAttribute("cy", point.y); circle.setAttribute("r", "7");
    const tooltip = document.createElementNS(SVG_NS, "title");
    tooltip.textContent = `${displayClass(point.event.predicted_class)} · ${point.event.confidence_bucket}`;
    circle.append(tooltip); group.append(circle);
    if (index >= attackPoints.length - 4) {
      const label = document.createElementNS(SVG_NS, "text");
      label.setAttribute("x", Math.min(900, Math.max(36, point.x)).toFixed(1));
      label.setAttribute("y", Math.max(20, point.y - 14));
      label.textContent = displayClass(point.event.predicted_class);
      group.append(label);
    }
    $("detection-points").append(group);
  });
  $("chart-window-count").textContent = `${samples.length} điểm`;
  updateAlertBanner(samples[samples.length - 1]);
}

function eventNode(event) {
  const row = document.createElement("div"); row.className = `event${event.is_alert ? " alert" : (event.predicted_class !== "Benign" ? " detection" : "")}`;
  const mark = document.createElement("i"); mark.className = "event-mark";
  const main = document.createElement("div"); main.className = "event-main";
  const label = document.createElement("strong"); label.textContent = displayClass(event.predicted_class);
  const policyState = event.is_alert
    ? `cảnh báo ${event.severity}`
    : (event.predicted_class === "Benign"
      ? "không cảnh báo"
      : (event.alert_decision_source === "trusted-shadow" ? "shadow · cảnh báo theo trusted head" : "dưới ngưỡng chính sách"));
  const detail = document.createElement("small"); detail.textContent = `Fusion 6 head · trusted ${event.client_id}: ${displayClass(event.trusted_predicted_class || event.predicted_class)} · ${policyState}`; main.append(label, detail);
  const meta = document.createElement("div"); meta.className = "event-meta"; meta.textContent = `${event.confidence_bucket}\n${event.inference_latency_ms} mili giây`;
  row.append(mark, main, meta); return row;
}

function renderHeadDiagnostics(event) {
  const entries = Object.entries(event?.head_predictions || {});
  if (!entries.length) {
    $("head-disagreement").textContent = "Chưa có dữ liệu";
    $("head-grid").innerHTML = '<div class="empty-event">Đang chờ dự đoán từ cả sáu head.</div>';
    return;
  }
  $("head-disagreement").textContent = `${event.head_disagreement_count}/6 head khác quyết định fusion`;
  $("head-grid").replaceChildren(...entries.map(([head, prediction]) => {
    const card = document.createElement("article");
    const isTrusted = head === event.client_id;
    const agrees = prediction.predicted_label === event.predicted_class;
    card.className = `head-card${isTrusted ? " trusted" : ""}${agrees ? " agrees" : " disagrees"}`;
    const name = document.createElement("small"); name.textContent = `HEAD ${head}${isTrusted ? " · TRUSTED" : ""}`;
    const label = document.createElement("strong"); label.textContent = displayClass(prediction.predicted_label);
    const confidence = document.createElement("span"); confidence.textContent = `Tin cậy ${prediction.confidence_bucket}`;
    card.append(name, label, confidence);
    return card;
  }));
}

async function pollMonitor() {
  try {
    const body = await json(`/api/monitor?after=${state.cursor}&limit=100`);
    state.cursor = body.next_cursor;
    if (body.events.length) {
      const fresh = body.events;
      state.chartEvents = [...state.chartEvents, ...fresh].slice(-80);
      state.events = [...fresh.slice().reverse(), ...state.events].slice(0, 100);
      $("event-list").replaceChildren(...state.events.map(eventNode));
      const latest = state.events[0];
      $("prediction-focus").querySelector("strong").textContent = displayClass(latest.predicted_class);
      $("prediction-focus").querySelector("span").textContent = `${latest.sensor_id} → đầu FedPer ${latest.client_id} · độ tin cậy ${latest.confidence_bucket} · entropy ${latest.entropy_bucket}`;
      renderHeadDiagnostics(latest);
      renderChart();
    }
    const metrics = body.metrics; $("metric-windows").textContent = metrics.windows; $("metric-alerts").textContent = metrics.events;
    $("metric-latency").textContent = metrics.inference_latency_ms_p95 == null ? "—" : `${Number(metrics.inference_latency_ms_p95).toFixed(1)} mili giây`;
    $("metric-drop").textContent = `${(Number(metrics.dropped_flows) / Math.max(1, Number(metrics.observations)) * 100).toFixed(2)}%`;
    $("system-dot").className = "status-dot ready"; $("system-label").textContent = "Luồng xử lý sẵn sàng";
  } catch (_) { $("system-dot").className = "status-dot error"; $("system-label").textContent = "Mất kết nối giám sát"; }
}

function clearMonitor() {
  state.events = [];
  state.chartEvents = [];
  $("event-list").innerHTML = '<div class="empty-event">Đã xóa màn hình; con trỏ dữ liệu vẫn được giữ.</div>';
  $("prediction-focus").querySelector("strong").textContent = "Đang chờ flow…";
  $("prediction-focus").querySelector("span").textContent = "Kịch bản lưu lượng và dự đoán của model là hai thông tin độc lập.";
  renderHeadDiagnostics(null);
  renderChart();
}

async function boot() {
  renderChart();
  try {
    [state.catalog, state.replay] = await Promise.all([json("/api/config"), json("/api/scientific-replay")]);
    renderCatalog(state.catalog); renderReplayCatalog(state.replay);
    $("system-dot").className = "status-dot ready"; $("system-label").textContent = "Bảng điều khiển sẵn sàng";
  } catch (error) { $("system-dot").className = "status-dot error"; $("system-label").textContent = error.message; }
  $("start-button").addEventListener("click", startRun); $("stop-button").addEventListener("click", stopRun);
  $("clear-events").addEventListener("click", clearMonitor);
  setInterval(pollRun, 1000); setInterval(pollMonitor, 1000); await pollRun(); await pollMonitor();
}
boot();
