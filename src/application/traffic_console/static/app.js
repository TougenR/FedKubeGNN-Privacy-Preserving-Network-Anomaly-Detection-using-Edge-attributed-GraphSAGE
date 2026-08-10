const state = {catalog: null, profile: null, run: null, busy: false, terminalSnapshot: null};
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
function updateActions() {
  $("start").disabled = state.busy || active() || !state.profile?.execution_enabled;
  $("stop").disabled = state.busy || !active();
  document.querySelectorAll(".profile").forEach((button) => { button.disabled = state.busy || active(); });
  [$("events"), $("interval")].forEach((input) => { input.disabled = state.busy || active() || !state.profile?.execution_enabled; });
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
function selectedControls() {
  return {events: Number($("events").value), interval_ms: Number($("interval").value)};
}
function renderTerminalCommand() {
  if (!state.profile) return;
  const controls = selectedControls();
  $("terminal-command").textContent = [
    "curl --silent --show-error --request POST \u005c",
    `  http://127.0.0.1:8090/api/runs/${state.profile.id} \u005c`,
    "  --header 'Content-Type: application/json' \u005c",
    `  --data '${JSON.stringify(controls)}'`,
  ].join("\n");
}
function appendTerminalLine(kind, message) {
  const output = $("terminal-output");
  const line = document.createElement("div");
  line.className = `terminal-line ${kind}`;
  line.textContent = message;
  output.append(line);
  while (output.children.length > 12) output.firstElementChild.remove();
  output.scrollTop = output.scrollHeight;
}
function resetTerminal(profile) {
  $("terminal-output").replaceChildren();
  state.terminalSnapshot = null;
  appendTerminalLine("info", `[profile] ${profile.reference_class} · mechanism=${profile.mechanism}`);
  appendTerminalLine("muted", `[policy] fixed target-group=${profile.target_group} · destination=tcp/${profile.destination_port}`);
  appendTerminalLine("muted", "[boundary] target, port, payload và model output không thể nhập từ terminal");
}
function selectProfile(profile) {
  state.profile = profile;
  document.querySelectorAll(".profile").forEach((button) => button.classList.toggle("selected", button.dataset.id === profile.id));
  $("profile-title").textContent = profile.reference_class;
  $("profile-status").textContent = scientificLabel(profile.scientific_status);
  $("profile-status").className = profile.execution_enabled ? "enabled" : "blocked";
  const observed = profile.expected_observables;
  $("profile-detail").className = "detail";
  $("profile-detail").innerHTML = "";
  [
    ["Cơ chế", profile.mechanism], ["Target group", profile.target_group],
    ["Cổng đích", profile.destination_port], ["Protocol", observed.protocols.join(", ")],
    ["Service", observed.services.join(", ")], ["Connection state", observed.connection_states.join(", ")]
  ].forEach(([label, value]) => {
    const cell = document.createElement("span"); const key = document.createElement("small"); const output = document.createElement("strong");
    key.textContent = label; output.textContent = String(value); cell.append(key, output); $("profile-detail").append(cell);
  });
  const note = document.createElement("p"); note.textContent = observed.note; $("profile-detail").append(note);
  $("events").value = profile.events; $("events").min = profile.controls.events.minimum; $("events").max = profile.controls.events.maximum;
  $("interval").value = profile.interval_ms; $("interval").min = profile.controls.interval_ms.minimum;
  resetTerminal(profile);
  refreshRate(); updateActions();
}
function renderCatalog(catalog) {
  state.catalog = catalog; $("profiles").replaceChildren();
  catalog.profiles.forEach((profile) => {
    const button = document.createElement("button"); button.className = `profile${profile.execution_enabled ? "" : " unavailable"}`; button.dataset.id = profile.id;
    const name = document.createElement("strong"); name.textContent = profile.reference_class;
    const meta = document.createElement("small"); meta.textContent = `${profile.mechanism} · :${profile.destination_port}`;
    button.append(name, meta); button.addEventListener("click", () => selectProfile(profile)); $("profiles").append(button);
  });
  const first = catalog.profiles.find((profile) => profile.execution_enabled) || catalog.profiles[0]; if (first) selectProfile(first);
}
function renderRun(record) {
  state.run = record;
  const collector = record?.pipeline?.collector || {};
  $("run-status").textContent = statusLabel(record?.status);
  $("run-status").className = active() ? "running" : (record?.status === "failed" ? "failed" : "");
  $("sent").textContent = `${record?.succeeded || 0}/${record?.attempted || 0}`;
  $("received").textContent = `${collector.accepted || 0}/${collector.received || 0}`;
  $("dropped").textContent = `${collector.late_dropped || 0}/${collector.duplicates || 0}`;
  $("failed").textContent = String(collector.processing_failures || 0);
  $("run-id").textContent = record ? `${record.profile_id} · ${record.run_id}` : "Không có run đang hoạt động";
  const showRecord = record && (active() || record.profile_id === state.profile?.id);
  if (showRecord) {
    const snapshot = [record.run_id, record.status, record.attempted || 0, record.succeeded || 0, collector.accepted || 0, collector.late_dropped || 0, collector.processing_failures || 0].join(":");
    if (snapshot !== state.terminalSnapshot) {
      state.terminalSnapshot = snapshot;
      if (["waiting-for-release", "running"].includes(record.status)) {
        appendTerminalLine("send", `[send] run=${record.run_id} · sent=${record.succeeded || 0}/${record.attempted || 0} · accepted=${collector.accepted || 0}`);
      } else if (record.status === "completed") {
        appendTerminalLine("success", `[done] sent=${record.succeeded || 0}/${record.attempted || 0} · accepted=${collector.accepted || 0} · drop=${collector.late_dropped || 0} · error=${collector.processing_failures || 0}`);
      } else if (record.status === "cancelled") {
        appendTerminalLine("warning", `[stop] run=${record.run_id} đã được dừng`);
      } else if (record.status === "failed") {
        appendTerminalLine("error", `[error] run=${record.run_id} thất bại`);
      }
    }
  }
  updateActions();
}
async function boot() {
  try {
    const [config, catalog] = await Promise.all([json("/api/config"), json("/api/profiles")]);
    const identity = config.identity;
    $("generator-name").textContent = identity.generator_name; $("generator-address").textContent = `${identity.generator_source_ipv4} · ${identity.generator_zone}`;
    $("target-name").textContent = identity.target_name; $("target-address").textContent = identity.target_ipv4;
    $("sensor-id").textContent = `sensor ${identity.sensor_id}`;
    renderCatalog(catalog); $("agent-dot").className = "ready"; $("agent-label").textContent = "Traffic agent ready"; await poll();
  } catch (error) { $("agent-dot").className = "error"; $("agent-label").textContent = error.message; }
}
async function start() {
  if (!state.profile?.execution_enabled || active()) return; state.busy = true; updateActions();
  appendTerminalLine("command", `[exec] ${state.profile.reference_class} · collector gate đang đăng ký`);
  try {
    const record = await json(`/api/runs/${state.profile.id}`, {method: "POST", body: JSON.stringify(selectedControls())});
    appendTerminalLine("success", `[gate] registered · run=${record.run_id} · agent released`);
    renderRun(record);
  }
  catch (error) { appendTerminalLine("error", `[error] ${error.message}`); $("agent-dot").className = "error"; $("agent-label").textContent = error.message; }
  finally { state.busy = false; updateActions(); }
}
async function stop() {
  if (!active()) return; state.busy = true; updateActions();
  appendTerminalLine("warning", `[exec] stop run=${state.run.run_id}`);
  try { const body = await json("/api/runs/current", {method: "DELETE"}); renderRun(body.run || null); }
  catch (error) { appendTerminalLine("error", `[error] ${error.message}`); $("agent-dot").className = "error"; $("agent-label").textContent = error.message; }
  finally { state.busy = false; updateActions(); }
}
async function poll() {
  if (state.busy || !state.catalog) return;
  try { const body = await json("/api/runs/current"); renderRun(body.run || null); $("agent-dot").className = "ready"; $("agent-label").textContent = active() ? "Đang phát traffic" : "Traffic agent ready"; }
  catch (error) { $("agent-dot").className = "error"; $("agent-label").textContent = error.message; }
}

$("events").addEventListener("input", refreshRate); $("interval").addEventListener("input", refreshRate);
$("start").addEventListener("click", start); $("stop").addEventListener("click", stop);
boot(); setInterval(poll, 1000);
