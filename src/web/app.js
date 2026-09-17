const els = {
  apiKey: document.getElementById("apiKey"),
  orgId: document.getElementById("orgId"),
  threadId: document.getElementById("threadId"),
  btnNew: document.getElementById("btnNew"),
  btnRefresh: document.getElementById("btnRefresh"),
  btnSend: document.getElementById("btnSend"),
  messageInput: document.getElementById("messageInput"),
  messages: document.getElementById("messages"),
  stateDump: document.getElementById("stateDump"),
  checkpointDump: document.getElementById("checkpointDump"),
  optionList: document.getElementById("optionList"),
  roundBadge: document.getElementById("roundBadge"),
};

let currentState = null;
const selectedSymptoms = new Set();

const API_KEY_STORAGE = "b2b_api_key";

function currentApiKey() {
  return (els.apiKey && els.apiKey.value.trim()) || localStorage.getItem(API_KEY_STORAGE) || "";
}

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  const key = currentApiKey();
  if (key) headers["X-API-Key"] = key;
  const res = await fetch(path, { ...options, headers });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = (data.error && data.error.message) || data.detail || res.statusText;
    throw new Error(msg);
  }
  return data;
}

function renderMessages(messages = []) {
  els.messages.innerHTML = "";
  for (const item of messages) {
    const div = document.createElement("div");
    const role = item.role === "human" || item.role === "user" ? "user" : "ai";
    div.className = `msg ${role}`;
    div.textContent = item.content || "";
    els.messages.appendChild(div);
  }
  els.messages.scrollTop = els.messages.scrollHeight;
}

function currentOptions(state) {
  const round = state.major_round || 1;
  if (round === 1) return state.body_part_options || [];
  if (round === 2) return state.disease_options || [];
  return state.symptom_options || [];
}

function isMultiSelect(state) {
  const round = state.major_round || 1;
  return (round === 3 || round === 4) && (state.symptom_options || []).length > 0;
}

function renderOptions(state) {
  currentState = state;
  selectedSymptoms.clear();
  els.roundBadge.textContent = `第 ${state.major_round || "-"} 轮`;
  els.optionList.innerHTML = "";

  const options = currentOptions(state);
  const multi = isMultiSelect(state);

  if (multi) {
    const submit = document.createElement("button");
    submit.className = "primary option-submit";
    submit.textContent = "提交已选症状";
    submit.onclick = () => step({ selected_symptoms: [...selectedSymptoms] });
    els.optionList.appendChild(submit);
  }

  for (const label of options) {
    const btn = document.createElement("button");
    btn.className = "option-btn";
    btn.textContent = label;
    btn.onclick = () => {
      if (multi) {
        btn.classList.toggle("active");
        if (selectedSymptoms.has(label)) selectedSymptoms.delete(label);
        else selectedSymptoms.add(label);
        return;
      }
      const round = state.major_round || 1;
      if (round === 1) step({ selected_body_part: label });
      else if (round === 2) step({ selected_disease: label });
    };
    els.optionList.appendChild(btn);
  }
}

function dashTitle(title) {
  return `------ ${title} ------`;
}

function formatValue(value) {
  if (value === undefined || value === null || value === "") return "（空）";
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value);
}

function formatCheckpoint(cp) {
  if (!cp || typeof cp !== "object") return "（无 checkpoint）";
  const ch = cp.channel_values || {};
  const msgs = Array.isArray(ch.messages) ? ch.messages : [];
  const dialog = msgs.length
    ? msgs.map((m) => `[${m.role || "ai"}]\n${m.content || ""}`).join("\n\n")
    : "（尚无对话）";
  const spell = ch.spell_correction;
  const spellText = spell
    ? [
        `来源: ${formatValue(spell.source)}`,
        `置信度: ${formatValue(spell.confidence)}`,
        `说明: ${formatValue(spell.explanation)}`,
      ].join("\n")
    : "（本轮未跑纠错）";
  const slots = ch.slots || {};
  return [
    dashTitle("会话"),
    `thread_id: ${formatValue(cp.thread_id)}`,
    `checkpoint_id: ${formatValue(cp.checkpoint_id)}`,
    `step: ${formatValue(cp.step)}`,
    `next: ${formatValue(cp.next)}`,
    "",
    dashTitle("对话"),
    dialog,
    "",
    dashTitle("纠错"),
    `raw_query: ${formatValue(ch.raw_query)}`,
    `query: ${formatValue(ch.query)}`,
    spellText,
    "",
    dashTitle("意图"),
    `来源: ${formatValue(ch.text_intent_source)}`,
    `模型: ${String(ch.text_intent_reason || "").includes("qwen-turbo") ? "qwen-turbo" : (ch.text_intent_source === "bert" ? "BERT" : "（空）")}`,
    `text_intent: ${formatValue(ch.text_intent)}`,
    `置信度: ${formatValue(ch.text_intent_confidence)}`,
    `原因: ${formatValue(ch.text_intent_reason)}`,
    "",
    dashTitle("轮次与锁定"),
    `major_round: ${formatValue(ch.major_round)}`,
    `session_mode: ${formatValue(ch.session_mode)}`,
    `next_action: ${formatValue(ch.next_action)}`,
    `body_part: ${formatValue(ch.body_part)}`,
    `disease_name: ${formatValue(ch.disease_name)}`,
    "",
    dashTitle("症状与证型"),
    `collected_symptoms: ${formatValue(ch.collected_symptoms)}`,
    `pending_current_main_hits: ${formatValue(ch.pending_current_main_hits)}`,
    `rollback_candidate_diseases: ${formatValue(ch.rollback_candidate_diseases)}`,
    `rollback_decision: ${formatValue(ch.rollback_decision)}`,
    `symptom_options: ${formatValue(ch.symptom_options)}`,
    `candidate_syndromes: ${formatValue(ch.candidate_syndromes)}`,
    "",
    dashTitle("槽位填充"),
    `来源: ${formatValue(slots.fill_source)}`,
    `模型: ${formatValue(slots.fill_model)}`,
    `原因: ${formatValue(slots.fill_reason)}`,
    `schema: ${formatValue(slots.schema)}`,
    `filled: ${formatValue(slots.filled)}`,
    `missing: ${formatValue(slots.missing)}`,
    `抽槽原文 span: ${formatValue(slots.extracted)}`,
    `入槽 mentions: ${formatValue(slots.mentions || slots.entities)}`,
    `body_part: ${formatValue(slots.body_part)}`,
    `disease_name: ${formatValue(slots.disease_name)}`,
    `symptoms: ${formatValue(slots.symptoms)}`,
    `entry_nodes: ${formatValue(slots.entry_nodes)}`,
    `alignments: ${formatValue(slots.alignments)}`,
    "",
    dashTitle("否定判定"),
    `策略: ${formatValue((slots.negation || {}).入槽策略)}`,
    `判定来源: ${formatValue((slots.negation || {}).judge)}`,
    `模型: ${formatValue((slots.negation || {}).qwen_model)}`,
    `qwen 判定: ${formatValue((slots.negation || {}).qwen_items)}`,
    `否定标记: ${formatValue((slots.negation || {}).markers)}`,
    `丢弃未入槽: ${formatValue((slots.negation || {}).dropped)}`,
    `保留入槽: ${formatValue((slots.negation || {}).kept)}`,
  ].join("\n");
}

function renderState(data) {
  currentState = data;
  els.threadId.value = data.thread_id || "";
  renderMessages(data.messages || []);
  renderOptions(data);
  const overview = {
      ok: data.ok,
      event: data.event,
      error: data.error,
      waiting_for_user: data.waiting_for_user,
      major_round: data.major_round,
      session_mode: data.session_mode,
      body_part: data.body_part,
      disease_name: data.disease_name,
      text_intent: data.text_intent,
      text_intent_reason: data.text_intent_reason,
      spell_correction: data.spell_correction_detail || data.spell_correction,
      messages: data.messages || [],
      collected_symptoms: data.collected_symptoms || [],
      candidate_syndromes: (data.candidate_syndromes || []).map((s) => s.syndrome_name),
      matched_syndrome: data.matched_syndrome,
      final_syndromes: (data.final_syndromes || []).map((s) => s.syndrome_name),
      auxiliary_round: data.auxiliary_round,
      paused_nodes: data.paused_nodes,
    };
  const checkpoint = data.checkpoint || {};
  els.stateDump.textContent = JSON.stringify(overview, null, 2);
  const cpEl = document.getElementById("checkpointDump");
  const cpText = formatCheckpoint(checkpoint);
  if (cpEl) {
    cpEl.textContent = cpText;
  } else {
    els.stateDump.textContent += "\n\n--- checkpoint ---\n" + cpText;
  }
}

async function createSession() {
  const data = await api("/api/sessions", {
    method: "POST",
    body: JSON.stringify({ org_id: els.orgId.value.trim() || null }),
  });
  renderState(data);
}

async function step(body) {
  const threadId = els.threadId.value.trim();
  if (!threadId) throw new Error("请先新建会话");
  const data = await api(`/api/sessions/${encodeURIComponent(threadId)}/step`, {
    method: "POST",
    body: JSON.stringify({ ...body, org_id: els.orgId.value.trim() || null }),
  });
  renderState(data);
}

function sendMessage() {
  const message = els.messageInput.value.trim();
  if (!message) return;
  els.messageInput.value = "";
  const round = currentState?.major_round || 1;
  if (round >= 3) {
    step({ message, selected_symptoms: [...selectedSymptoms, message] }).catch((err) => alert(err.message));
    return;
  }
  step({ message }).catch((err) => alert(err.message));
}

if (els.apiKey) {
  els.apiKey.value = localStorage.getItem(API_KEY_STORAGE) || "";
  els.apiKey.addEventListener("change", () => {
    localStorage.setItem(API_KEY_STORAGE, els.apiKey.value.trim());
  });
}

els.btnNew.onclick = () => createSession().catch((err) => alert(err.message));
els.btnRefresh.onclick = async () => {
  const threadId = els.threadId.value.trim();
  if (!threadId) return;
  renderState(await api(`/api/sessions/${encodeURIComponent(threadId)}`));
};
els.btnSend.onclick = sendMessage;
els.messageInput.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.isComposing || event.keyCode === 229) return;
  event.preventDefault();
  sendMessage();
});

createSession().catch(() => {
  els.stateDump.textContent = "点击「新建会话」开始";
  if (els.checkpointDump) els.checkpointDump.textContent = "-";
});
