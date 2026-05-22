function el(tag, className, text) {
  const n = document.createElement(tag);
  if (className) n.className = className;
  if (text !== undefined) n.textContent = text;
  return n;
}

function scrollToBottom(container) {
  container.scrollTop = container.scrollHeight;
}

function addMessage(container, kind, text, meta) {
  const bubble = el("div", `msg msg--${kind}`);
  bubble.textContent = text;
  if (meta) {
    const metaDiv = el("div", "msg__meta");
    metaDiv.appendChild(meta);
    bubble.appendChild(metaDiv);
  }
  container.appendChild(bubble);
  scrollToBottom(container);
}

function renderResponseMeta(resp) {
  const wrap = el("div");

  if (resp.alerts && resp.alerts.length) {
    const chips = el("div", "chips");
    resp.alerts.forEach((a) => {
      chips.appendChild(el("div", "chip", `ALERTA: ${a.message || ""}`));
    });
    wrap.appendChild(chips);
  }

  if (resp.sources && resp.sources.length) {
    const chips = el("div", "chips");
    resp.sources.forEach((s, idx) => {
      const title = s.title || s.doc_id || `Fonte ${idx + 1}`;
      chips.appendChild(el("div", "chip", `[${idx + 1}] ${title}`));
    });
    wrap.appendChild(chips);
  }

  if (resp.request_id) {
    wrap.appendChild(el("div", "", `request_id: ${resp.request_id}`));
  }

  return wrap;
}

async function sendMessage() {
  const messages = document.getElementById("messages");
  const input = document.getElementById("messageInput");
  const sendBtn = document.getElementById("sendBtn");
  const patientSelect = document.getElementById("patientSelect");
  const clinicianId = document.getElementById("clinicianId");
  const status = document.getElementById("chatStatus");

  const text = (input.value || "").trim();
  if (!text) return;

  addMessage(messages, "out", text);
  input.value = "";
  sendBtn.disabled = true;
  status.textContent = "processando...";

  try {
    const payload = {
      message: text,
      patient_id: patientSelect.value || null,
      clinician_id: (clinicianId.value || "").trim() || null,
    };
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const t = await res.text();
      addMessage(messages, "in", `Erro (${res.status}): ${t}`);
      return;
    }
    const data = await res.json();
    const meta = renderResponseMeta(data);
    addMessage(messages, "in", data.answer || "", meta);
  } catch (e) {
    addMessage(messages, "in", `Falha ao comunicar com o servidor: ${String(e)}`);
  } finally {
    sendBtn.disabled = false;
    status.textContent = "online";
  }
}

function bootstrap() {
  const input = document.getElementById("messageInput");
  const sendBtn = document.getElementById("sendBtn");

  sendBtn.addEventListener("click", sendMessage);
  input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") sendMessage();
  });
}

bootstrap();
