const state = {catalog: null, profile: null, run: null, busy: false, timer: null};
const $ = (id) => document.getElementById(id);

async function json(url, options = {}) {
  const response = await fetch(url, {headers: {"Content-Type": "application/json"}, ...options});
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
  return body;
}

function active() { return ["waiting-for-release", "running"].includes(state.run?.status); }
function statusLabel(value) {
  return ({"waiting-for-release": "Chờ pipeline", running: "Đang phát", completed: "Hoàn tất", cancelled: "Đã dừng", failed: "Thất bại"})[value] || value || "Chưa chạy";
}
function scientificLabel(value) {
  return ({candidate: "Candidate", "control-not-class-equivalent": "Benign control", "blocked-target-not-ready": "Không khả dụng", "unsupported-dataset-artifact": "Không khả dụng"})[value] || value;
}
function selectedControls() { return {events: Number($("events").value), interval_ms: Number($("interval").value)}; }
function updateActions() {
  $("start").disabled = state.busy || active() || !state.profile?.execution_enabled;
  $("stop").disabled = state.busy || !active();
  document.querySelectorAll(".profile").forEach((button) => { button.disabled = state.busy || active(); });
  [$("events"), $("interval")].forEach((input) => { input.disabled = state.busy || active() || !state.profile?.execution_enabled; });
}
function renderTerminalCommand() {
  if (!state.profile) return;
  $("terminal-command").textContent = [
    "curl --silent --show-error --request POST " + String.fromCharCode(92),
    `  http://127.0.0.1:8090/api/runs/${state.profile.id} ` + String.fromCharCode(92),
    "  --header 'Content-Type: application/json' " + String.fromCharCode(92),
    `  --data '${JSON.stringify(selectedControls())}'`,
  ].join("\n");
}
function refreshRate() {
  if (!state.profile) return;
  const events = Number($("events").value || state.profile.events);
  const configuredMax = Number(state.profile.controls.interval_ms.maximum);
  const scheduleMax = events > 1 ? Math.floor(120000 / (events - 1)) : configuredMax;
  const effectiveMax = Math.min(configuredMax, scheduleMax);
  $("interval").max = String(effectiveMax);
  if (Number($("interval").value) > effectiveMax) $("interval").value = String(effectiveMax);
  $("rate").textContent = `${(1000 / Math.max(1, Number($("interval").value))).toFixed(2)} event/s`;
  renderTerminalCommand();
}

function renderExecution(record) {
  const output = $("terminal-output"); output.replaceChildren();
  if (!record) {
    output.innerHTML = '<div class="terminal-line muted">[idle] Chưa có event thực thi.</div>';
    return;
  }
  const header = document.createElement("div");
  header.className = `terminal-line ${record.status === "failed" ? "error" : active() ? "warning" : "success"}`;
  header.textContent = `[run] ${record.run_id} · ${statusLabel(record.status)} · ${record.succeeded || 0}/${record.attempted || 0} sent`;
  output.append(header);
  (record.execution_evidence || []).forEach((item) => {
    const line = document.createElement("div"); line.className = `terminal-line ${item.success ? "send" : "error"}`;
    const flags = item.tcp_flags ? ` flags=${item.tcp_flags}` : "";
    line.textContent = `[${String(item.event_index).padStart(2, "0")}] ${new Date(item.timestamp * 1000).toISOString().slice(11, 23)} ${item.source} → ${item.target}:${item.port}${flags} · ${item.action} · ${item.success ? "SENT" : "FAILED"}`;
    output.append(line);
  });
  output.scrollTop = output.scrollHeight;
}

function normalize(value) { return value === "-" ? "unknown" : String(value); }
function listComparison(label, expected, observed) {
  if (!expected || observed === null || observed === undefined) return {label, expected: expected?.join(", ") || "N/A", observed: "N/A", status: "na"};
  const match = expected.map(normalize).includes(normalize(observed));
  return {label, expected: expected.join(", "), observed: String(observed), status: match ? "match" : "mismatch"};
}
function rangeComparison(label, expected, observed) {
  if (!expected || observed === null || observed === undefined) return {label, expected: expected ? `${expected.minimum}–${expected.maximum} ${expected.unit}` : "N/A", observed: "N/A", status: "na"};
  const value = Number(observed); const width = Math.max(1, expected.maximum - expected.minimum);
  const match = value >= expected.minimum && value <= expected.maximum;
  const near = value >= expected.minimum - width * 0.2 && value <= expected.maximum + width * 0.2;
  return {label, expected: `${expected.minimum}–${expected.maximum} ${expected.unit}`, observed: `${value} ${expected.unit}`, status: match ? "match" : near ? "near" : "mismatch"};
}
function renderZeek(record) {
  const evidence = record?.pipeline?.zeek_evidence || [];
  const evidenceProfile = state.catalog?.profiles.find((profile) => profile.id === record?.profile_id) || state.profile;
  const latest = evidence[evidence.length - 1] || null; const expected = evidenceProfile?.expected_observables;
  const rows = expected ? [
    listComparison("Protocol", expected.protocols, latest?.protocol),
    listComparison("Service", expected.services, latest?.service),
    listComparison("Conn state", expected.connection_states, latest?.connection_state),
    listComparison("History", expected.histories, latest?.history),
    listComparison("Response", expected.response_behaviors, latest?.response_behavior),
    rangeComparison("Orig packets", expected.orig_packets, latest?.orig_packets),
    rangeComparison("Resp packets", expected.resp_packets, latest?.resp_packets),
    rangeComparison("Orig bytes", expected.orig_bytes, latest?.orig_bytes),
    rangeComparison("Resp bytes", expected.resp_bytes, latest?.resp_bytes),
    rangeComparison("Density", expected.flow_density, latest?.density),
  ] : [];
  $("zeek-compare").replaceChildren(...rows.map((row) => {
    const cell = document.createElement("div"); cell.className = `comparison ${row.status}`;
    cell.innerHTML = `<small>${row.label}</small><span><b>IoT-23</b> ${row.expected}</span><span><b>Zeek</b> ${row.observed}</span>`; return cell;
  }));
  const log = $("zeek-log"); log.replaceChildren();
  if (!evidence.length) { log.innerHTML = '<div class="empty-log">Chưa có conn.log cho run này.</div>'; return; }
  evidence.forEach((item) => {
    const line = document.createElement("div"); line.className = "conn-line";
    line.textContent = `${new Date(item.timestamp * 1000).toISOString()} ${item.source} → ${item.target}:${item.port} ${item.protocol} ${item.service} ${item.connection_state} ${item.history} pkts=${item.orig_packets ?? "-"}/${item.resp_packets ?? "-"} bytes=${item.orig_bytes ?? "-"}/${item.resp_bytes ?? "-"}`;
    log.append(line);
  });
  log.scrollTop = log.scrollHeight;
}

const countWords = {agent: "sent", zeek: "observed", shipper: "delivered", gateway: "received", collector: "accepted", window: "windowed", inference: "inferred", router: "stored"};
function renderPipeline(record) {
  const stages = record?.pipeline?.stages || [];
  document.querySelectorAll("[data-stage]").forEach((card) => {
    const stage = stages.find((item) => item.key === card.dataset.stage);
    card.className = stage?.status || "idle";
    card.querySelector("span").textContent = `${stage?.count || 0} ${countWords[card.dataset.stage]}`;
  });
}

function selectProfile(profile) {
  state.profile = profile;
  document.querySelectorAll(".profile").forEach((button) => button.classList.toggle("selected", button.dataset.id === profile.id));
  $("profile-title").textContent = profile.reference_class;
  const targets = profile.fixed_targets.map((item) => `${item.alias}=${item.endpoint}`).join(" · ");
  $("profile-summary").innerHTML = `<span><small>CƠ CHẾ</small><b>${profile.mechanism}</b></span><span><small>ĐÍCH CỐ ĐỊNH</small><b>${targets}</b></span><span><small>PORT</small><b>tcp/${profile.destination_port}</b></span><span><small>REFERENCE</small><b>${scientificLabel(profile.scientific_status)} · n=${profile.expected_observables.reference_support || "N/A"}</b></span>`;
  $("mechanism-detail").textContent = `executor: ${profile.mechanism} · ${targets} · không gọi hping3/nmap`;
  $("events").value = profile.events; $("events").min = profile.controls.events.minimum; $("events").max = profile.controls.events.maximum;
  $("interval").value = profile.interval_ms; $("interval").min = profile.controls.interval_ms.minimum;
  refreshRate(); renderZeek(state.run); updateActions();
}
function renderCatalog(catalog) {
  state.catalog = catalog; $("profiles").replaceChildren();
  catalog.profiles.forEach((profile) => {
    const button = document.createElement("button"); button.className = `profile${profile.execution_enabled ? "" : " unavailable"}`; button.dataset.id = profile.id;
    button.innerHTML = `<strong>${profile.reference_class}</strong><small>${profile.mechanism} · :${profile.destination_port}</small>`;
    button.addEventListener("click", () => selectProfile(profile)); $("profiles").append(button);
  });
  const first = catalog.profiles.find((profile) => profile.execution_enabled) || catalog.profiles[0]; if (first) selectProfile(first);
}
function renderRun(record) {
  state.run = record; renderPipeline(record); renderExecution(record); renderZeek(record);
  if (record) {
    $("run-badge").textContent = `${active() ? "ACTIVE RUN" : "LAST RUN"} · ${record.profile_id} · ${record.run_id}`;
    $("run-badge").className = active() ? "running" : record.status === "failed" ? "failed" : "last";
    $("run-id").textContent = `${record.profile_id} · ${statusLabel(record.status)}`;
  } else { $("run-badge").textContent = "NO RUN"; $("run-badge").className = ""; $("run-id").textContent = "Chưa chạy"; }
  updateActions();
}
function switchTab(name) {
  ["agent", "zeek"].forEach((tab) => {
    const selected = tab === name; $(`tab-${tab}`).classList.toggle("active", selected); $(`tab-${tab}`).setAttribute("aria-selected", String(selected)); $(`panel-${tab}`).classList.toggle("active", selected);
  });
}
async function boot() {
  try {
    const [config, catalog] = await Promise.all([json("/api/config"), json("/api/profiles")]); const identity = config.identity;
    $("generator-name").textContent = identity.generator_name; $("generator-address").textContent = `${identity.generator_source_ipv4} · ${identity.generator_zone}`;
    $("target-name").textContent = identity.target_name; $("target-address").textContent = identity.target_ipv4; $("sensor-id").textContent = `sensor ${identity.sensor_id}`;
    renderCatalog(catalog); $("agent-dot").className = "ready"; $("agent-label").textContent = "Traffic agent ready"; await poll();
  } catch (error) { $("agent-dot").className = "error"; $("agent-label").textContent = error.message; schedulePoll(); }
}
async function start() {
  if (!state.profile?.execution_enabled || active()) return; state.busy = true; updateActions();
  try { renderRun(await json(`/api/runs/${state.profile.id}`, {method: "POST", body: JSON.stringify(selectedControls())})); }
  catch (error) { $("agent-dot").className = "error"; $("agent-label").textContent = error.message; }
  finally { state.busy = false; updateActions(); schedulePoll(); }
}
async function stop() {
  if (!active()) return; state.busy = true; updateActions();
  try { const body = await json("/api/runs/current", {method: "DELETE"}); renderRun(body.run || null); }
  catch (error) { $("agent-dot").className = "error"; $("agent-label").textContent = error.message; }
  finally { state.busy = false; updateActions(); schedulePoll(); }
}
function schedulePoll() { clearTimeout(state.timer); state.timer = setTimeout(poll, active() ? 500 : 2000); }
async function poll() {
  if (state.busy || !state.catalog) { schedulePoll(); return; }
  try { const body = await json("/api/runs/current"); renderRun(body.run || null); $("agent-dot").className = "ready"; $("agent-label").textContent = active() ? "Đang phát traffic" : "Traffic agent ready"; }
  catch (error) { $("agent-dot").className = "error"; $("agent-label").textContent = error.message; }
  finally { schedulePoll(); }
}

$("events").addEventListener("input", refreshRate); $("interval").addEventListener("input", refreshRate);
$("start").addEventListener("click", start); $("stop").addEventListener("click", stop);
$("tab-agent").addEventListener("click", () => switchTab("agent")); $("tab-zeek").addEventListener("click", () => switchTab("zeek"));
boot();
