const state = { catalog: null, selected: null, cursor: 0, events: [] };
const $ = (id) => document.getElementById(id);
const classHelp = {
  "Benign": "Flow được model xem là bình thường.",
  "Attack": "Nhãn tấn công tổng quát trong dataset.",
  "C&C": "Hành vi command-and-control đã biết trong IoT-23.",
  "C&C-HeartBeat": "Nhịp liên lạc C&C lặp lại trong benchmark.",
  "DDoS": "Flow thuộc lớp distributed denial-of-service.",
  "Okiru": "Họ hành vi Okiru trong tập huấn luyện.",
  "PartOfAHorizontalPortScan": "Flow thuộc horizontal port-scan benchmark."
};

async function json(url, options = {}) {
  const response = await fetch(url, { headers: {"Content-Type": "application/json"}, ...options });
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
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
    const label = document.createElement("label"); label.textContent = name.replaceAll("_", " ");
    const hint = document.createElement("small"); hint.textContent = `${bound.minimum}–${bound.maximum} ${bound.unit}`; label.append(hint);
    const input = document.createElement("input"); input.type = "number"; input.name = name; input.min = bound.minimum; input.max = bound.maximum; input.value = scenario.defaults[name]; input.required = true;
    row.append(label, input); $("parameter-form").append(row);
  });
  $("start-button").disabled = false;
}

function renderCatalog(catalog) {
  $("disclaimer").textContent = catalog.disclaimer;
  catalog.scenarios.forEach((scenario) => {
    const card = document.createElement("button"); card.type = "button"; card.className = "scenario-card"; card.dataset.id = scenario.id;
    const title = document.createElement("strong"); title.textContent = scenario.display_name;
    const summary = document.createElement("span"); summary.textContent = scenario.summary;
    card.append(title, summary); card.addEventListener("click", () => selectScenario(scenario)); $("scenario-grid").append(card);
  });
  catalog.model_classes.forEach((name) => {
    const card = document.createElement("article"); card.className = "model-class";
    const title = document.createElement("strong"); title.textContent = name;
    const help = document.createElement("small"); help.textContent = classHelp[name] || "Model output class.";
    card.append(title, help); $("class-list").append(card);
  });
}

function runParameters() {
  return Object.fromEntries([...new FormData($("parameter-form")).entries()].map(([key, value]) => [key, Number(value)]));
}

async function startRun(event) {
  event.preventDefault();
  if (!state.selected) return;
  try {
    await json("/api/runs", { method: "POST", body: JSON.stringify({scenario_id: state.selected.id, parameters: runParameters()}) });
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
    $("run-status").textContent = run.status.toUpperCase(); $("run-id").textContent = run.run_id;
    $("attempted").textContent = run.attempted; $("succeeded").textContent = run.succeeded; $("failed").textContent = run.failed;
    $("progress-bar").style.width = `${Math.min(100, run.attempted / Math.max(1, expectedAttempts(run)) * 100)}%`;
    const active = run.status === "running"; $("start-button").disabled = active || !state.selected; $("stop-button").disabled = !active;
  } catch (_) { /* health indicator owns connectivity state */ }
}

function eventNode(event) {
  const row = document.createElement("div"); row.className = `event${event.is_alert ? " alert" : ""}`;
  const mark = document.createElement("i"); mark.className = "event-mark";
  const main = document.createElement("div"); main.className = "event-main";
  const label = document.createElement("strong"); label.textContent = event.predicted_class;
  const detail = document.createElement("small"); detail.textContent = `${event.sensor_id} → head ${event.client_id} · ${event.is_alert ? event.severity : "no alert"}`; main.append(label, detail);
  const meta = document.createElement("div"); meta.className = "event-meta"; meta.textContent = `${event.confidence_bucket}\n${event.inference_latency_ms} ms`;
  row.append(mark, main, meta); return row;
}

async function pollMonitor() {
  try {
    const body = await json(`/api/monitor?after=${state.cursor}&limit=100`);
    state.cursor = body.next_cursor;
    if (body.events.length) {
      state.events = [...body.events.reverse(), ...state.events].slice(0, 100);
      $("event-list").replaceChildren(...state.events.map(eventNode));
      const latest = state.events[0];
      $("prediction-focus").querySelector("strong").textContent = latest.predicted_class;
      $("prediction-focus").querySelector("span").textContent = `${latest.sensor_id} → FedPer head ${latest.client_id} · confidence ${latest.confidence_bucket} · entropy ${latest.entropy_bucket}`;
    }
    const metrics = body.metrics; $("metric-windows").textContent = metrics.windows; $("metric-alerts").textContent = metrics.events;
    $("metric-latency").textContent = metrics.inference_latency_ms_p95 == null ? "—" : `${Number(metrics.inference_latency_ms_p95).toFixed(1)} ms`;
    $("metric-drop").textContent = `${(Number(metrics.dropped_flows) / Math.max(1, Number(metrics.observations)) * 100).toFixed(2)}%`;
    $("system-dot").className = "status-dot ready"; $("system-label").textContent = "Pipeline sẵn sàng";
  } catch (_) { $("system-dot").className = "status-dot error"; $("system-label").textContent = "Monitor mất kết nối"; }
}

async function boot() {
  try { state.catalog = await json("/api/config"); renderCatalog(state.catalog); $("system-dot").className = "status-dot ready"; $("system-label").textContent = "Console sẵn sàng"; }
  catch (error) { $("system-dot").className = "status-dot error"; $("system-label").textContent = error.message; }
  $("start-button").addEventListener("click", startRun); $("stop-button").addEventListener("click", stopRun);
  $("clear-events").addEventListener("click", () => { state.events = []; $("event-list").innerHTML = '<div class="empty-event">Đã xóa màn hình; cursor vẫn được giữ.</div>'; });
  setInterval(pollRun, 1000); setInterval(pollMonitor, 1000); await pollRun(); await pollMonitor();
}
boot();
