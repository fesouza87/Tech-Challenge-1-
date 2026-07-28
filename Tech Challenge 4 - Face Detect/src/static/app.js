const state = {
  selectedPatientId: null,
  patients: [],
  audioPreviewUrl: null,
  activeTab: "pipelines",
  liveAlerts: [],
  notificationOpen: false,
};

const DEMO_VITAL_PATH = "c:\\Users\\felip\\source\\FIAP\\TechChallenge1\\Tech-Challenge-1-\\Tech Challenge 4 - Face Detect\\vital\\0001.vital";

const elements = {
  patientsList: document.getElementById("patients-list"),
  liveAlerts: document.getElementById("live-alerts"),
  recentEvents: document.getElementById("recent-events"),
  recentAlerts: document.getElementById("recent-alerts"),
  riskSummary: document.getElementById("risk-summary"),
  clinicalBanner: document.getElementById("clinical-banner"),
  transcriptPanel: document.getElementById("transcript-panel"),
  vitalsMonitor: document.getElementById("vitals-monitor"),
  patientEvents: document.getElementById("patient-events"),
  patientAlerts: document.getElementById("patient-alerts"),
  selectedPatientLabel: document.getElementById("selected-patient-label"),
  alertsBellButton: document.getElementById("alerts-bell-button"),
  alertsBellCount: document.getElementById("alerts-bell-count"),
  alertsPopup: document.getElementById("alerts-popup"),
  alertsPopupClose: document.getElementById("alerts-popup-close"),
  connectionBadge: document.getElementById("connection-badge"),
  lastRefresh: document.getElementById("last-refresh"),
  refreshButton: document.getElementById("refresh-button"),
  audioUploadForm: document.getElementById("audio-upload-form"),
  audioPatientId: document.getElementById("audio-patient-id"),
  audioLanguage: document.getElementById("audio-language"),
  audioTimestamp: document.getElementById("audio-timestamp"),
  audioFile: document.getElementById("audio-file"),
  audioSubmitButton: document.getElementById("audio-submit-button"),
  audioDemoButton: document.getElementById("audio-demo-button"),
  audioResetButton: document.getElementById("audio-reset-button"),
  audioPreview: document.getElementById("audio-preview"),
  audioUploadStatus: document.getElementById("audio-upload-status"),
  audioUploadResult: document.getElementById("audio-upload-result"),
  vitalImportForm: document.getElementById("vital-import-form"),
  vitalPatientId: document.getElementById("vital-patient-id"),
  vitalIntervalSeconds: document.getElementById("vital-interval-seconds"),
  vitalMaxSamples: document.getElementById("vital-max-samples"),
  vitalFilePath: document.getElementById("vital-file-path"),
  vitalFileUpload: document.getElementById("vital-file-upload"),
  vitalSubmitButton: document.getElementById("vital-submit-button"),
  vitalDemoButton: document.getElementById("vital-demo-button"),
  vitalResetButton: document.getElementById("vital-reset-button"),
  vitalUploadStatus: document.getElementById("vital-upload-status"),
  vitalUploadResult: document.getElementById("vital-upload-result"),
};

const tabButtons = Array.from(document.querySelectorAll("[data-tab-target]"));
const tabPanels = Array.from(document.querySelectorAll("[data-tab-panel]"));
const VITAL_CARD_MEANINGS = {
  SpO2: "Saturacao periferica de oxigenio",
  FC: "Frequencia cardiaca",
  PAS: "Pressao arterial sistolica",
  PAD: "Pressao arterial diastolica",
  FR: "Frequencia respiratoria",
  Temp: "Temperatura corporal",
};

async function fetchJson(url, options = undefined) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`Falha ao carregar ${url}`);
  }
  return response.json();
}

async function loadOverview() {
  const overview = await fetchJson("/api/dashboard/overview");
  state.patients = overview.patients || [];

  elements.lastRefresh.textContent = `Atualizado em ${new Date().toLocaleTimeString("pt-BR")}`;
  state.liveAlerts = overview.recent_alerts || [];

  renderPatients(state.patients);
  renderRecentEvents(overview.recent_events || []);
  renderRecentAlerts(overview.recent_alerts || []);
  renderNotificationAlerts(state.liveAlerts);

  if (!state.selectedPatientId && state.patients.length > 0) {
    state.selectedPatientId = state.patients[0].patient_id;
  }
  if (state.selectedPatientId) {
    await loadPatient(state.selectedPatientId);
  }
}

async function loadPatient(patientId) {
  const payload = await fetchJson(`/api/dashboard/patient/${patientId}`);
  state.selectedPatientId = patientId;
  elements.selectedPatientLabel.textContent = `Paciente ${patientId}`;
  renderRiskSummary(payload.risk_summary);
  renderClinicalBanner(payload.risk_summary, payload.latest_vitals, payload.alerts || []);
  renderVitalsMonitor(payload.latest_vitals);
  renderPatientEvents(payload.events || []);
  renderPatientAlerts(payload.alerts || []);
  renderTranscriptPanel(payload.events || []);
  renderPatients(state.patients);
}

function setActiveTab(tabName) {
  state.activeTab = tabName;

  tabButtons.forEach((button) => {
    const isActive = button.dataset.tabTarget === tabName;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-selected", isActive ? "true" : "false");
  });

  tabPanels.forEach((panel) => {
    const isActive = panel.dataset.tabPanel === tabName;
    panel.classList.toggle("is-active", isActive);
    panel.hidden = !isActive;
  });
}

function initializeTabs() {
  tabButtons.forEach((button) => {
    button.addEventListener("click", () => setActiveTab(button.dataset.tabTarget));
  });
  setActiveTab(state.activeTab);
}

function renderPatients(items) {
  if (!items.length) {
    elements.patientsList.innerHTML = `<div class="empty-state">Nenhum paciente processado ate o momento.</div>`;
    return;
  }

  elements.patientsList.innerHTML = items
    .map((item) => {
      const selected = item.patient_id === state.selectedPatientId;
      return `
        <div class="list-item" ${selected ? 'style="border-color:#38bdf8;"' : ""}>
          <button type="button" data-patient-id="${item.patient_id}">
            <div class="list-title">
              <span>${item.patient_id}</span>
              <span class="badge ${severityBadgeClass(item.highest_severity)}">${item.highest_severity}</span>
            </div>
            <div class="list-meta">Eventos: ${item.event_count} | Alertas: ${item.alert_count}</div>
            <div class="list-meta">Modalidades: ${(item.active_modalities || []).join(", ") || "n/a"}</div>
            <div class="list-meta">Ultimo sinal: ${item.latest_signal || "n/a"}</div>
          </button>
        </div>
      `;
    })
    .join("");

  elements.patientsList.querySelectorAll("button[data-patient-id]").forEach((button) => {
    button.addEventListener("click", async () => {
      await loadPatient(button.dataset.patientId);
      setActiveTab("paciente");
    });
  });
}

function renderRecentEvents(items) {
  renderGenericList(
    elements.recentEvents,
    items,
    (item) => `
      <div class="list-title">
        <span>${item.modality.toUpperCase()} | ${item.patient_id}</span>
        <span class="badge ${severityBadgeClass(item.severity)}">${item.severity}</span>
      </div>
      <div class="list-meta">${item.signal}</div>
      <div class="list-meta">Score: ${formatScore(item.anomaly_score)} | ${formatDate(item.timestamp)}</div>
    `,
    "Nenhum evento recente."
  );
}

function renderRecentAlerts(items) {
  renderGenericList(
    elements.recentAlerts,
    items,
    (item) => `
      <div class="list-title">
        <span>${item.title}</span>
        <span class="badge ${severityBadgeClass(item.severity)}">${item.severity}</span>
      </div>
      <div class="list-meta">${item.patient_id} | ${item.modality}</div>
      <div class="list-meta">${item.message}</div>
    `,
    "Nenhum alerta recente."
  );
}

function renderRiskSummary(summary) {
  elements.riskSummary.innerHTML = `
    <div class="info-grid">
      <div class="info-item"><strong>Paciente</strong>${summary.patient_id}</div>
      <div class="info-item"><strong>Severidade</strong><span class="badge ${severityBadgeClass(summary.highest_severity)}">${summary.highest_severity}</span></div>
      <div class="info-item"><strong>Eventos</strong>${summary.event_count}</div>
      <div class="info-item"><strong>Alertas</strong>${summary.alert_count}</div>
      <div class="info-item"><strong>Score Medio</strong>${formatScore(summary.average_anomaly_score)}</div>
      <div class="info-item"><strong>Ultimo Sinal</strong>${summary.latest_signal || "n/a"}</div>
    </div>
    <div class="tag-row">
      ${(summary.active_modalities || []).map((modality) => `<span class="tag">${modality}</span>`).join("")}
    </div>
  `;
}

function renderClinicalBanner(summary, latestVitals, alerts) {
  const alert = (alerts || [])[0];
  const vitalsEvent = latestVitals?.event;
  const latest = latestVitals?.details?.latest_sample || {};
  const recommendation = alert?.recommended_action || recommendationFromSignal(vitalsEvent?.signal);
  const headline = vitalsEvent ? describeSignal(vitalsEvent.signal) : "Nenhuma anomalia fisiologica recente";

  elements.clinicalBanner.className = `detail-card clinical-banner ${severityBannerClass(summary.highest_severity)}`;
  elements.clinicalBanner.innerHTML = `
    <div class="clinical-banner-header">
      <div>
        <p class="eyebrow">Resumo Clinico</p>
        <h3>${headline}</h3>
        <p class="muted">${recommendation}</p>
      </div>
      <div class="clinical-banner-status">
        <span class="badge ${severityBadgeClass(summary.highest_severity)}">${(summary.highest_severity || "info").toUpperCase()}</span>
        <span class="muted">Ultimo sinal: ${summary.latest_signal || "n/a"}</span>
      </div>
    </div>
    <div class="clinical-banner-grid">
      <div class="clinical-chip">
        <strong>SpO2 Atual</strong>
        <span>${formatVitalValue(latest.spo2, "%")}</span>
      </div>
      <div class="clinical-chip">
        <strong>FC Atual</strong>
        <span>${formatVitalValue(latest.heart_rate, "bpm")}</span>
      </div>
      <div class="clinical-chip">
        <strong>FR Atual</strong>
        <span>${formatVitalValue(latest.respiratory_rate, "irpm")}</span>
      </div>
      <div class="clinical-chip">
        <strong>Risco</strong>
        <span>${(summary.highest_severity || "info").toUpperCase()}</span>
      </div>
    </div>
  `;
}

function renderTranscriptPanel(events) {
  const transcriptEvent = events.find((item) => item.transcript_excerpt);
  if (!transcriptEvent) {
    elements.transcriptPanel.innerHTML = `<div class="empty-state">Nenhuma transcricao disponivel para este paciente.</div>`;
    return;
  }

  const evidence = transcriptEvent.evidence || [];
  const metadata = transcriptEvent.metadata || {};
  elements.transcriptPanel.innerHTML = `
    <div class="info-grid">
      <div class="info-item"><strong>Origem</strong>${metadata.transcript_source || "request"}</div>
      <div class="info-item"><strong>Modalidade</strong>${transcriptEvent.modality}</div>
      <div class="info-item"><strong>Score</strong>${formatScore(transcriptEvent.anomaly_score)}</div>
      <div class="info-item"><strong>Severidade</strong><span class="badge ${severityBadgeClass(transcriptEvent.severity)}">${transcriptEvent.severity}</span></div>
    </div>
    <p>${escapeHtml(transcriptEvent.transcript_excerpt)}</p>
    <ul class="evidence-list">
      ${evidence.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
    </ul>
  `;
}

function renderVitalsMonitor(latestVitals) {
  if (!latestVitals || !latestVitals.details) {
    elements.vitalsMonitor.className = "detail-card empty-state";
    elements.vitalsMonitor.innerHTML = "Nenhum processamento de sinais vitais disponivel para este paciente.";
    return;
  }

  const details = latestVitals.details;
  const latest = details.latest_sample || {};
  const series = details.sample_series || [];
  const cards = [
    buildVitalCard("SpO2", "spo2", latest.spo2, "%", series, 92, null),
    buildVitalCard("FC", "heart_rate", latest.heart_rate, "bpm", series, 45, 120),
    buildVitalCard("PAS", "systolic_bp", latest.systolic_bp, "mmHg", series, 90, 180),
    buildVitalCard("PAD", "diastolic_bp", latest.diastolic_bp, "mmHg", series, 60, 110),
    buildVitalCard("FR", "respiratory_rate", latest.respiratory_rate, "irpm", series, 10, 24),
    buildVitalCard("Temp", "temperature_c", latest.temperature_c, "C", series, 36, 38.2),
  ];

  elements.vitalsMonitor.className = "detail-card vitals-monitor-shell";
  elements.vitalsMonitor.innerHTML = `
    <div class="vitals-monitor-grid">
      ${cards.join("")}
    </div>
    <div class="vitals-monitor-footer">
      <span class="tag">arquivo=${escapeHtml(details.vital_file_path || "demo")}</span>
      <span class="tag">janela=${escapeHtml(details.selected_sample_count || series.length || "n/a")} pontos</span>
      <span class="tag">intervalo=${escapeHtml(details.interval_seconds || "n/a")}s</span>
    </div>
  `;
}

function renderPatientEvents(events) {
  renderGenericList(
    elements.patientEvents,
    events,
    (item) => `
      <div class="list-title">
        <span>${item.modality.toUpperCase()}</span>
        <span class="badge ${severityBadgeClass(item.severity)}">${item.severity}</span>
      </div>
      <div class="list-meta">${item.signal}</div>
      <div class="list-meta">Score: ${formatScore(item.anomaly_score)} | ${formatDate(item.timestamp)}</div>
    `,
    "Nenhum evento encontrado."
  );
}

function renderPatientAlerts(alerts) {
  renderGenericList(
    elements.patientAlerts,
    alerts,
    (item) => `
      <div class="list-title">
        <span>${item.title}</span>
        <span class="badge ${severityBadgeClass(item.severity)}">${item.severity}</span>
      </div>
      <div class="list-meta">${item.message}</div>
      <div class="list-meta">Acao: ${item.recommended_action}</div>
    `,
    "Nenhum alerta para este paciente."
  );
}

function renderGenericList(target, items, template, emptyMessage) {
  if (!items.length) {
    target.innerHTML = `<div class="empty-state">${emptyMessage}</div>`;
    return;
  }
  target.innerHTML = items.map((item) => `<div class="list-item">${template(item)}</div>`).join("");
}

function connectAlertStream() {
  const source = new EventSource("/api/alerts/stream");
  source.addEventListener("open", () => {
    elements.connectionBadge.textContent = "Stream conectado";
    elements.connectionBadge.className = "badge badge-info";
  });
  source.addEventListener("alert", async (event) => {
    const alert = JSON.parse(event.data);
    prependLiveAlert(alert);
    await loadOverview();
  });
  source.addEventListener("error", () => {
    elements.connectionBadge.textContent = "Stream indisponivel";
    elements.connectionBadge.className = "badge badge-high";
  });
}

function prependLiveAlert(alert) {
  state.liveAlerts = [alert, ...state.liveAlerts].slice(0, 12);
  renderNotificationAlerts(state.liveAlerts);
}

function renderNotificationAlerts(items) {
  const alerts = items || [];
  elements.alertsBellCount.textContent = String(alerts.length);
  elements.alertsBellCount.hidden = alerts.length === 0;

  renderGenericList(
    elements.liveAlerts,
    alerts,
    (item) => `
      <div class="list-title">
        <span>${item.title}</span>
        <span class="badge ${severityBadgeClass(item.severity)}">${item.severity}</span>
      </div>
      <div class="list-meta">${item.patient_id} | ${item.modality}</div>
      <div class="list-meta">${item.message}</div>
      <div class="list-meta">${formatDate(item.created_at)}</div>
    `,
    "Nenhum alerta ao vivo ate o momento."
  );
}

function setNotificationOpen(isOpen) {
  state.notificationOpen = isOpen;
  elements.alertsPopup.hidden = !isOpen;
  elements.alertsBellButton.setAttribute("aria-expanded", isOpen ? "true" : "false");
}

function toggleNotificationPopup() {
  setNotificationOpen(!state.notificationOpen);
}

function handleDocumentClick(event) {
  if (!state.notificationOpen) {
    return;
  }

  const clickedInsidePopup = elements.alertsPopup.contains(event.target);
  const clickedBell = elements.alertsBellButton.contains(event.target);
  if (!clickedInsidePopup && !clickedBell) {
    setNotificationOpen(false);
  }
}

function handleDocumentKeydown(event) {
  if (event.key === "Escape" && state.notificationOpen) {
    setNotificationOpen(false);
  }
}

async function submitPreparedAudioForm() {
  setUploadStatus("Enviando audio para processamento...", "info");
  const formData = buildAudioFormData();
  const response = await fetch("/api/pipelines/audio/upload", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await extractErrorMessage(response));
  }

  const result = await response.json();
  state.selectedPatientId = result.event.patient_id;
  elements.audioPatientId.value = result.event.patient_id;
  renderUploadResult(result);
  setUploadStatus("Audio processado com sucesso.", "success");
  await loadOverview();
  await loadPatient(result.event.patient_id);
}

async function processCurrentAudioForm() {
  setUploadBusy(true);
  try {
    await submitPreparedAudioForm();
  } catch (error) {
    console.error(error);
    setUploadStatus(error.message || "Falha ao processar audio.", "error");
  } finally {
    setUploadBusy(false);
  }
}

async function handleAudioUpload(event) {
  event.preventDefault();
  await processCurrentAudioForm();
}

function buildAudioFormData() {
  const formData = new FormData(elements.audioUploadForm);
  const timestampValue = elements.audioTimestamp.value;
  const file = elements.audioFile.files[0];

  formData.set("patient_id", elements.audioPatientId.value.trim());
  formData.set("language", elements.audioLanguage.value.trim() || "pt-BR");

  if (timestampValue) {
    formData.set("timestamp", new Date(timestampValue).toISOString());
  } else {
    formData.delete("timestamp");
  }

  ["pause_ratio", "speech_rate_wpm", "vocal_energy", "articulation_clarity", "breathing_irregularity"].forEach((name) => {
    const value = formData.get(name);
    if (value === "" || value === null) {
      formData.delete(name);
    }
  });

  if (!file) {
    throw new Error("Selecione um arquivo de audio ou use o botao Rodar Audio Demo.");
  }

  // `audio_file`   
  formData.set("audio_file", file, file.name);

  return formData;
}

function renderUploadResult(result) {
  const event = result.event || {};
  const details = result.details || {};
  const alert = result.generated_alert;
  const risk = result.patient_risk || {};
  const transcript = event.transcript_excerpt || "Sem transcricao retornada.";

  elements.audioUploadResult.innerHTML = `
    <div class="info-grid">
      <div class="info-item"><strong>Paciente</strong>${event.patient_id || "n/a"}</div>
      <div class="info-item"><strong>Pipeline</strong>${result.pipeline || "audio"}</div>
      <div class="info-item"><strong>Origem Transcricao</strong>${event.metadata?.transcript_source || details.transcript_source || "request"}</div>
      <div class="info-item"><strong>Score</strong>${formatScore(event.anomaly_score)}</div>
      <div class="info-item"><strong>Severidade</strong><span class="badge ${severityBadgeClass(event.severity)}">${event.severity || "info"}</span></div>
      <div class="info-item"><strong>Risco Atual</strong><span class="badge ${severityBadgeClass(risk.highest_severity)}">${risk.highest_severity || "info"}</span></div>
    </div>
    <p><strong>Transcricao:</strong> ${escapeHtml(transcript)}</p>
    <ul class="evidence-list">
      ${(event.evidence || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
    </ul>
    <div class="tag-row">
      <span class="tag">speech=${details.azure_speech_success ? "ok" : "off"}</span>
      <span class="tag">text=${details.azure_text_success ? "ok" : "off"}</span>
      <span class="tag">source=${escapeHtml(details.source || "upload")}</span>
      ${details.uploaded_filename ? `<span class="tag">${escapeHtml(details.uploaded_filename)}</span>` : ""}
      ${alert ? `<span class="tag">${escapeHtml(alert.title)}</span>` : ""}
    </div>
  `;
}

function renderVitalResult(result) {
  const event = result.event || {};
  const details = result.details || {};
  const alert = result.generated_alert;
  const latest = details.latest_sample || {};
  const risk = result.patient_risk || {};

  elements.vitalUploadResult.innerHTML = `
    <div class="info-grid">
      <div class="info-item"><strong>Paciente</strong>${event.patient_id || "n/a"}</div>
      <div class="info-item"><strong>Pipeline</strong>${result.pipeline || "vitals"}</div>
      <div class="info-item"><strong>Amostras</strong>${details.selected_sample_count || latest.sample_count || "n/a"}</div>
      <div class="info-item"><strong>Score</strong>${formatScore(event.anomaly_score)}</div>
      <div class="info-item"><strong>Severidade</strong><span class="badge ${severityBadgeClass(event.severity)}">${event.severity || "info"}</span></div>
      <div class="info-item"><strong>Risco Atual</strong><span class="badge ${severityBadgeClass(risk.highest_severity)}">${risk.highest_severity || "info"}</span></div>
    </div>
    <p><strong>Arquivo:</strong> ${escapeHtml(details.vital_file_path || DEMO_VITAL_PATH)}</p>
    <p><strong>Ultima amostra:</strong> HR ${latest.heart_rate ?? "n/a"} | SpO2 ${latest.spo2 ?? "n/a"} | PAS ${latest.systolic_bp ?? "n/a"} | PAD ${latest.diastolic_bp ?? "n/a"} | FR ${latest.respiratory_rate ?? "n/a"} | Temp ${latest.temperature_c ?? "n/a"}</p>
    <ul class="evidence-list">
      ${(event.evidence || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
    </ul>
    <div class="tag-row">
      <span class="tag">source=${escapeHtml(details.source || "vitaldb_import")}</span>
      <span class="tag">interval=${escapeHtml(details.interval_seconds || elements.vitalIntervalSeconds.value || "60")}s</span>
      ${alert ? `<span class="tag">${escapeHtml(alert.title)}</span>` : ""}
    </div>
  `;
}

function setUploadBusy(isBusy) {
  elements.audioSubmitButton.disabled = isBusy;
  elements.audioDemoButton.disabled = isBusy;
  elements.audioResetButton.disabled = isBusy;
  elements.audioFile.disabled = isBusy;
}

function setVitalBusy(isBusy) {
  elements.vitalSubmitButton.disabled = isBusy;
  elements.vitalDemoButton.disabled = isBusy;
  elements.vitalResetButton.disabled = isBusy;
  elements.vitalFilePath.disabled = isBusy;
  elements.vitalFileUpload.disabled = isBusy;
  elements.vitalIntervalSeconds.disabled = isBusy;
  elements.vitalMaxSamples.disabled = isBusy;
}

function setUploadStatus(message, kind) {
  const classes = {
    info: "badge-info",
    success: "badge-low",
    error: "badge-high",
  };
  elements.audioUploadStatus.className = "detail-card compact-state";
  elements.audioUploadStatus.innerHTML = `
    <div class="list-title">
      <span>Status Do Upload</span>
      <span class="badge ${classes[kind] || "badge-neutral"}">${kind}</span>
    </div>
    <div class="list-meta">${escapeHtml(message)}</div>
  `;
}

function setVitalStatus(message, kind) {
  const classes = {
    info: "badge-info",
    success: "badge-low",
    error: "badge-high",
  };
  elements.vitalUploadStatus.className = "detail-card compact-state";
  elements.vitalUploadStatus.innerHTML = `
    <div class="list-title">
      <span>Status Do VitalDB</span>
      <span class="badge ${classes[kind] || "badge-neutral"}">${kind}</span>
    </div>
    <div class="list-meta">${escapeHtml(message)}</div>
  `;
}

async function extractErrorMessage(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    const payload = await response.json();
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
    return JSON.stringify(payload.detail || payload);
  }
  return response.text();
}

function buildVitalImportPayload() {
  const payload = {
    patient_id: elements.vitalPatientId.value.trim(),
    interval_seconds: Number(elements.vitalIntervalSeconds.value || 60),
    max_samples: Number(elements.vitalMaxSamples.value || 24),
  };

  if (elements.vitalFilePath.value.trim()) {
    payload.vital_file_path = elements.vitalFilePath.value.trim();
  }

  return payload;
}

function buildVitalUploadFormData() {
  const formData = new FormData();
  const file = elements.vitalFileUpload.files[0];

  formData.set("patient_id", elements.vitalPatientId.value.trim());
  formData.set("interval_seconds", String(Number(elements.vitalIntervalSeconds.value || 60)));
  formData.set("max_samples", String(Number(elements.vitalMaxSamples.value || 24)));

  if (elements.vitalFilePath.value.trim()) {
    formData.set("vital_file_path", elements.vitalFilePath.value.trim());
  }

  if (file) {
    formData.set("vital_file", file, file.name);
  }

  return formData;
}

async function processVitalImport(payload, statusMessage) {
  setVitalBusy(true);
  setVitalStatus(statusMessage, "info");

  try {
    const response = await fetch("/api/pipelines/vitals/vitaldb", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      throw new Error(await extractErrorMessage(response));
    }

    const result = await response.json();
    state.selectedPatientId = result.event.patient_id;
    elements.vitalPatientId.value = result.event.patient_id;
    renderVitalResult(result);
    setVitalStatus("VitalDB processado com sucesso.", "success");
    await loadOverview();
    await loadPatient(result.event.patient_id);
  } catch (error) {
    console.error(error);
    setVitalStatus(error.message || "Falha ao processar o arquivo .vital.", "error");
  } finally {
    setVitalBusy(false);
  }
}

async function processVitalUpload(statusMessage) {
  setVitalBusy(true);
  setVitalStatus(statusMessage, "info");

  try {
    const response = await fetch("/api/pipelines/vitals/vitaldb/upload", {
      method: "POST",
      body: buildVitalUploadFormData(),
    });
    if (!response.ok) {
      throw new Error(await extractErrorMessage(response));
    }

    const result = await response.json();
    state.selectedPatientId = result.event.patient_id;
    elements.vitalPatientId.value = result.event.patient_id;
    renderVitalResult(result);
    setVitalStatus("VitalDB processado com sucesso.", "success");
    await loadOverview();
    await loadPatient(result.event.patient_id);
  } catch (error) {
    console.error(error);
    setVitalStatus(error.message || "Falha ao processar o arquivo .vital.", "error");
  } finally {
    setVitalBusy(false);
  }
}

async function loadDemoAudio() {
  setUploadBusy(true);
  setUploadStatus("Carregando audio demo...", "info");

  try {
    const response = await fetch(`/api/dashboard/demo-audio?ts=${Date.now()}`, {
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(await extractErrorMessage(response));
    }

    const blob = await response.blob();
    const file = new File([blob], "speech_demo.wav", { type: blob.type || "audio/wav" });
    const transfer = new DataTransfer();
    transfer.items.add(file);
    elements.audioFile.files = transfer.files;
    elements.audioPatientId.value = "PAZDEMO";
    elements.audioLanguage.value = "pt-BR";
    elements.audioTimestamp.value = defaultTimestampValue();
    handleAudioFileSelection();
    setUploadStatus("Audio demo carregado. Iniciando processamento...", "info");
    await submitPreparedAudioForm();
  } catch (error) {
    console.error(error);
    setUploadStatus(error.message || "Falha ao carregar audio demo.", "error");
  } finally {
    setUploadBusy(false);
  }
}

async function handleVitalImport(event) {
  event.preventDefault();
  if (elements.vitalFileUpload.files[0]) {
    await processVitalUpload("Enviando arquivo .vital...");
    return;
  }

  await processVitalImport(buildVitalImportPayload(), "Importando sinais vitais...");
}

async function loadDemoVital() {
  elements.vitalPatientId.value = "PVITAL01";
  elements.vitalIntervalSeconds.value = "60";
  elements.vitalMaxSamples.value = "24";
  elements.vitalFilePath.value = "";
  await processVitalImport(
    {
      patient_id: elements.vitalPatientId.value.trim(),
      interval_seconds: Number(elements.vitalIntervalSeconds.value),
      max_samples: Number(elements.vitalMaxSamples.value),
    },
    "Carregando demo VitalDB..."
  );
}

function handleAudioFileSelection() {
  const file = elements.audioFile.files[0];
  if (state.audioPreviewUrl) {
    URL.revokeObjectURL(state.audioPreviewUrl);
    state.audioPreviewUrl = null;
  }

  if (!file) {
    elements.audioPreview.removeAttribute("src");
    elements.audioPreview.load();
    return;
  }

  state.audioPreviewUrl = URL.createObjectURL(file);
  elements.audioPreview.src = state.audioPreviewUrl;
  elements.audioPreview.load();
}

function resetAudioForm() {
  elements.audioUploadForm.reset();
  elements.audioPatientId.value = state.selectedPatientId || "PAZ01";
  elements.audioLanguage.value = "pt-BR";
  elements.audioTimestamp.value = defaultTimestampValue();
  elements.audioUploadResult.innerHTML = "O resultado do ultimo processamento aparecera aqui.";
  setUploadStatus("Formulario limpo. Pronto para novo teste.", "info");
  handleAudioFileSelection();
}

function resetVitalForm() {
  elements.vitalImportForm.reset();
  elements.vitalPatientId.value = "PVITAL01";
  elements.vitalIntervalSeconds.value = "60";
  elements.vitalMaxSamples.value = "24";
  elements.vitalUploadResult.innerHTML = "O resultado do ultimo processamento aparecera aqui.";
  setVitalStatus("Formulario limpo. Pronto para novo teste com VitalDB.", "info");
}

function buildVitalCard(label, fieldName, currentValue, unit, series, lowThreshold, highThreshold) {
  const points = (series || [])
    .map((item) => normalizeNumber(item[fieldName]))
    .filter((value) => value !== null);
  const trend = computeTrend(points);
  const severity = vitalSeverity(currentValue, lowThreshold, highThreshold);
  const path = buildSparklinePath(points);
  const range = buildRangeLabel(points, unit);
  const meaning = VITAL_CARD_MEANINGS[label] || label;

  return `
    <article class="vital-card ${severityCardClass(severity)}">
      <div class="vital-card-header">
        <div class="vital-card-heading">
          <span class="vital-card-label">${label}</span>
          <span class="vital-card-meaning">${meaning}</span>
        </div>
        <span class="badge ${severityBadgeClass(severity)}">${trend.label}</span>
      </div>
      <div class="vital-card-value">${formatVitalValue(currentValue, unit)}</div>
      <div class="vital-card-range">${range}</div>
      <svg class="sparkline" viewBox="0 0 160 52" preserveAspectRatio="none" aria-hidden="true">
        <path d="${path}" />
      </svg>
    </article>
  `;
}

function normalizeNumber(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function computeTrend(values) {
  if (!values.length) {
    return { label: "sem dado", delta: 0 };
  }
  const first = values[0];
  const last = values[values.length - 1];
  const delta = last - first;
  if (Math.abs(delta) < 0.5) {
    return { label: "estavel", delta };
  }
  return { label: delta > 0 ? "subindo" : "caindo", delta };
}

function buildSparklinePath(values) {
  if (!values.length) {
    return "M0 40 L160 40";
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  return values
    .map((value, index) => {
      const x = (index / Math.max(values.length - 1, 1)) * 160;
      const y = 46 - ((value - min) / span) * 36;
      return `${index === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
}

function buildRangeLabel(values, unit) {
  if (!values.length) {
    return "Sem serie recente";
  }
  const min = Math.min(...values).toFixed(0);
  const max = Math.max(...values).toFixed(0);
  return `Faixa recente ${min}-${max}${unit ? ` ${unit}` : ""}`;
}

function vitalSeverity(value, lowThreshold, highThreshold) {
  const numeric = normalizeNumber(value);
  if (numeric === null) {
    return "info";
  }
  if (lowThreshold !== null && numeric < lowThreshold) {
    return "high";
  }
  if (highThreshold !== null && numeric > highThreshold) {
    return "high";
  }
  if (
    (lowThreshold !== null && numeric < lowThreshold + Math.max(lowThreshold * 0.05, 2)) ||
    (highThreshold !== null && numeric > highThreshold - Math.max(highThreshold * 0.05, 2))
  ) {
    return "medium";
  }
  return "low";
}

function severityCardClass(severity) {
  switch (severity) {
    case "high":
      return "vital-card-high";
    case "medium":
      return "vital-card-medium";
    case "low":
      return "vital-card-low";
    default:
      return "vital-card-info";
  }
}

function severityBannerClass(severity) {
  switch (severity) {
    case "high":
      return "clinical-banner-high";
    case "medium":
      return "clinical-banner-medium";
    case "low":
      return "clinical-banner-low";
    default:
      return "clinical-banner-info";
  }
}

function formatVitalValue(value, unit) {
  const numeric = normalizeNumber(value);
  if (numeric === null) {
    return "n/a";
  }
  const digits = unit === "C" ? 1 : 0;
  return `${numeric.toFixed(digits)}${unit ? ` ${unit}` : ""}`;
}

function describeSignal(signal) {
  switch (signal) {
    case "dessaturacao":
      return "Dessaturacao detectada na janela recente";
    case "instabilidade_hemodinamica":
      return "Instabilidade hemodinamica detectada";
    case "desvio_sinais_vitais":
      return "Desvio relevante nos sinais vitais";
    default:
      return "Paciente com monitorizacao estavel";
  }
}

function recommendationFromSignal(signal) {
  switch (signal) {
    case "dessaturacao":
      return "Conferir oxigenacao, ventilacao e tendencia respiratoria imediatamente.";
    case "instabilidade_hemodinamica":
      return "Revisar pressao arterial, perfusao e contexto assistencial.";
    case "desvio_sinais_vitais":
      return "Correlacionar sinais recentes com prescricao e contexto clinico.";
    default:
      return "Manter monitorizacao e observar novas alteracoes.";
  }
}

function severityBadgeClass(severity) {
  switch (severity) {
    case "high":
      return "badge-high";
    case "medium":
      return "badge-medium";
    case "low":
      return "badge-low";
    default:
      return "badge-info";
  }
}

function formatScore(value) {
  return Number(value || 0).toFixed(2);
}

function formatDate(value) {
  return new Date(value).toLocaleString("pt-BR");
}

function defaultTimestampValue() {
  const now = new Date();
  const offsetMs = now.getTimezoneOffset() * 60000;
  return new Date(now.getTime() - offsetMs).toISOString().slice(0, 16);
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

elements.refreshButton.addEventListener("click", () => loadOverview());
elements.alertsBellButton.addEventListener("click", toggleNotificationPopup);
elements.alertsPopupClose.addEventListener("click", () => setNotificationOpen(false));
elements.audioUploadForm.addEventListener("submit", handleAudioUpload);
elements.audioDemoButton.addEventListener("click", loadDemoAudio);
elements.audioFile.addEventListener("change", handleAudioFileSelection);
elements.audioResetButton.addEventListener("click", resetAudioForm);
elements.vitalImportForm.addEventListener("submit", handleVitalImport);
elements.vitalDemoButton.addEventListener("click", loadDemoVital);
elements.vitalResetButton.addEventListener("click", resetVitalForm);
elements.audioTimestamp.value = defaultTimestampValue();
setUploadStatus("Envie um arquivo de audio ou rode o demo para testar o pipeline.", "info");
setVitalStatus("Importe um .vital ou rode o demo para testar o pipeline de sinais vitais.", "info");
renderNotificationAlerts(state.liveAlerts);
initializeTabs();
document.addEventListener("click", handleDocumentClick);
document.addEventListener("keydown", handleDocumentKeydown);

loadOverview().catch((error) => {
  console.error(error);
  elements.patientsList.innerHTML = `<div class="empty-state">Falha ao carregar dados do dashboard.</div>`;
});
connectAlertStream();
