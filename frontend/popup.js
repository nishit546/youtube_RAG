/**
 * popup.js — YT RAG Assistant v2.0
 *
 * Handles:
 * - Auto-detecting video ID from active YouTube tab
 * - Multi-video management (load, switch, remove)
 * - Mode-aware /ask calls
 * - Cross-video /compare calls
 * - Source timestamps as clickable links
 * - Conversational memory (history sent with each request)
 * - Clear + re-ingest actions
 */

const API = "http://localhost:8000";

// ── DOM refs ─────────────────────────────────────────────────────────────────
const videoIdInput   = document.getElementById("videoIdInput");
const loadBtn        = document.getElementById("loadBtn");
const videoChipsEl   = document.getElementById("videoChips");
const modeRow        = document.getElementById("modeRow");
const chatEl         = document.getElementById("chat");
const welcomeEl      = document.getElementById("welcome");
const questionInput  = document.getElementById("questionInput");
const sendBtn        = document.getElementById("sendBtn");
const statusBar      = document.getElementById("statusBar");
const errorBar       = document.getElementById("errorBar");
const clearBtn       = document.getElementById("clearBtn");
const refreshBtn     = document.getElementById("refreshBtn");
const compareBtn     = document.getElementById("compareBtn");

// ── State ────────
let loadedVideos  = {};      // { video_id: { title, source } }
let activeVideoId = null;
let activeMode    = "ask";
let isLoading     = false;
let localHistory  = {};      // { video_id: [{question, answer}] }

// ── Init ──────────────────────────────────────────────────────────────────────
(async () => {
  // Try to auto-detect video ID from active YouTube tab
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab && tab.url && tab.url.includes("youtube.com/watch")) {
      const response = await chrome.tabs.sendMessage(tab.id, { type: "GET_VIDEO_ID" });
      if (response && response.videoId) {
        videoIdInput.value = response.videoId;
        setStatus("info", `Detected video: ${response.videoId}`);
      }
    }
  } catch (_) {
    // Not on a YouTube tab — fine, let user type manually
  }

  // Load previously ingested videos from server
  await loadVideoList();
})();

// ── Video list from server ────────────────────────────────────────────────────
async function loadVideoList() {
  try {
    const res = await fetch(`${API}/videos`);
    const data = await res.json();
    for (const v of data.videos || []) {
      loadedVideos[v.video_id] = { title: v.title, source: "cache" };
    }
    renderChips();
  } catch (_) { /* server not running */ }
}

// ── Load / ingest video ───────────────────────────────────────────────────────
loadBtn.addEventListener("click", () => ingestVideo(videoIdInput.value.trim()));

async function ingestVideo(videoId, forceRefresh = false) {
  if (!videoId) { showError("Please enter a YouTube video ID."); return; }
  if (isLoading) return;

  setLoading(true);
  setStatus("loading", `Ingesting video ${videoId}…`);
  hideError();

  try {
    const res = await fetch(`${API}/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: videoId, force_refresh: forceRefresh }),
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Ingestion failed.");
    }

    loadedVideos[data.video_id] = { title: data.title, source: data.source };
    activeVideoId = data.video_id;
    renderChips();

    const cacheLabel = data.cached ? " (from cache)" : ` — ${data.chunk_count} chunks embedded`;
    setStatus("ok", `Ready: "${data.title}"${cacheLabel}`);
    hideWelcome();
  } catch (err) {
    handleError(err);
  } finally {
    setLoading(false);
  }
}

// ── Video chips ───────────────────────────────────────────────────────────────
function renderChips() {
  videoChipsEl.innerHTML = "";
  for (const [vid, meta] of Object.entries(loadedVideos)) {
    const chip = document.createElement("div");
    chip.className = `video-chip${vid === activeVideoId ? " active" : ""}`;
    chip.innerHTML = `
      <span class="chip-dot"></span>
      <span class="chip-label" title="${meta.title}">${meta.title}</span>
      <span class="chip-close" data-vid="${vid}">✕</span>
    `;
    chip.addEventListener("click", (e) => {
      if (e.target.classList.contains("chip-close")) {
        removeVideo(e.target.dataset.vid);
      } else {
        setActiveVideo(vid);
      }
    });
    videoChipsEl.appendChild(chip);
  }

  // Show compare button if 2+ videos loaded
  if (Object.keys(loadedVideos).length >= 2) {
    compareBtn.classList.add("visible");
  } else {
    compareBtn.classList.remove("visible");
  }
}

function setActiveVideo(videoId) {
  activeVideoId = videoId;
  renderChips();
  setStatus("ok", `Active: "${loadedVideos[videoId]?.title || videoId}"`);
}

function removeVideo(videoId) {
  delete loadedVideos[videoId];
  if (activeVideoId === videoId) {
    activeVideoId = Object.keys(loadedVideos)[0] || null;
  }
  renderChips();
  if (!activeVideoId) showWelcome();
}

// ── Mode selector ─────────────────────────────────────────────────────────────
modeRow.querySelectorAll(".mode-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    modeRow.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    activeMode = btn.dataset.mode;
  });
});

// ── Ask / Send ────────────────────────────────────────────────────────────────
sendBtn.addEventListener("click", handleAsk);
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleAsk();
  }
});

// Auto-resize textarea
questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = Math.min(questionInput.scrollHeight, 100) + "px";
});

async function handleAsk() {
  const question = questionInput.value.trim();
  if (!question) return;
  if (!activeVideoId) { showError("Load a video first!"); return; }
  if (isLoading) return;

  appendUserMessage(question);
  questionInput.value = "";
  questionInput.style.height = "auto";
  setLoading(true);
  setStatus("loading", "Thinking…");
  hideError();

  try {
    const res = await fetch(`${API}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        video_id: activeVideoId,
        question,
        mode: activeMode,
        k: activeMode === "navigate" ? 8 : 5,
      }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed.");

    appendBotMessage(data.answer, data.sources, data.mode);

    // Keep local history for display purposes
    if (!localHistory[activeVideoId]) localHistory[activeVideoId] = [];
    localHistory[activeVideoId].push({ question, answer: data.answer });

    setStatus("ok", `Answered (${data.sources.length} source${data.sources.length !== 1 ? "s" : ""})`);
  } catch (err) {
    handleError(err);
    appendBotMessage(`⚠️ ${err.message}`, [], "error");
  } finally {
    setLoading(false);
  }
}

// ── Compare ───────────────────────────────────────────────────────────────────
compareBtn.addEventListener("click", async () => {
  const question = questionInput.value.trim() || "Compare these videos and highlight the main differences.";
  if (Object.keys(loadedVideos).length < 2) return;

  appendUserMessage(`[Compare] ${question}`);
  questionInput.value = "";
  setLoading(true);
  setStatus("loading", "Comparing videos…");
  hideError();

  try {
    const res = await fetch(`${API}/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        video_ids: Object.keys(loadedVideos),
        question,
        k_per_video: 3,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Compare failed.");

    appendBotMessage(data.answer, data.sources, "compare");
    setStatus("ok", `Compare done (${data.sources.length} chunks)`);
  } catch (err) {
    handleError(err);
    appendBotMessage(`⚠️ ${err.message}`, [], "error");
  } finally {
    setLoading(false);
  }
});

// ── Header actions ─────────────────────────────────────────────────────────────
clearBtn.addEventListener("click", async () => {
  if (!activeVideoId) return;
  try {
    await fetch(`${API}/history/${activeVideoId}`, { method: "DELETE" });
  } catch (_) {}
  localHistory[activeVideoId] = [];
  chatEl.innerHTML = "";
  hideError();
  setStatus("ok", "Chat cleared.");
});

refreshBtn.addEventListener("click", async () => {
  if (!activeVideoId) { showError("No active video to refresh."); return; }
  await ingestVideo(activeVideoId, true);
});

// ── Message rendering ─────────────────────────────────────────────────────────
function appendUserMessage(text) {
  hideWelcome();
  const div = document.createElement("div");
  div.className = "msg msg-user";
  div.innerHTML = `<div class="msg-bubble">${escapeHtml(text)}</div>`;
  chatEl.appendChild(div);
  scrollChat();
}

function appendBotMessage(text, sources = [], mode = "ask") {
  hideWelcome();
  const modeLabelMap = {
    ask: "💬 Answer", summary: "📝 Summary", keypoints: "🔑 Key Points",
    deep: "🔬 Deep Dive", quiz: "🧠 Quiz", navigate: "🧭 Navigation", compare: "⚡ Compare"
  };

  const div = document.createElement("div");
  div.className = "msg msg-bot";

  // Format text: preserve newlines and bullets
  const formatted = escapeHtml(text)
    .replace(/•\s/g, "• ")
    .replace(/\n/g, "<br/>");

  let sourcesHtml = "";
  if (sources && sources.length > 0) {
    const chips = sources.map(src =>
      `<a class="source-chip" href="${src.url}" target="_blank" title="${escapeHtml(src.text.slice(0, 120))}…">
        ▶ ${src.start_fmt} – ${src.end_fmt}
        ${src.title && src.title !== src.video_id ? `<span style="opacity:0.6;font-size:9px">[${escapeHtml(src.title.slice(0,20))}]</span>` : ""}
      </a>`
    ).join("");
    sourcesHtml = `<div class="sources">${chips}</div>`;
  }

  div.innerHTML = `
    <span class="msg-mode-tag">${modeLabelMap[mode] || mode}</span>
    <div class="msg-bubble">${formatted}${sourcesHtml}</div>
  `;
  chatEl.appendChild(div);
  scrollChat();
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function scrollChat() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function hideWelcome() {
  if (welcomeEl) welcomeEl.style.display = "none";
}

function showWelcome() {
  if (welcomeEl) welcomeEl.style.display = "";
}

function setLoading(loading) {
  isLoading = loading;
  sendBtn.disabled = loading;
  loadBtn.disabled = loading;
}

function setStatus(type, msg) {
  const icons = {
    loading: `<div class="spinner"></div>`,
    ok: `<div class="status-dot ok"></div>`,
    error: `<div class="status-dot error"></div>`,
    info: `<div class="status-dot warn"></div>`,
  };
  statusBar.innerHTML = `${icons[type] || ""} <span>${escapeHtml(msg)}</span>`;
}

function showError(msg) {
  errorBar.textContent = msg;
  errorBar.style.display = "block";
}

function hideError() {
  errorBar.style.display = "none";
}

function handleError(err) {
  let msg = "An unknown error occurred.";
  if (err.message?.includes("Failed to fetch") || err.name === "TypeError") {
    msg = "⚠️ Cannot reach backend — is the server running on port 8000?";
  } else if (err.message?.includes("not ingested")) {
    msg = "Video not loaded. Click Load first.";
  } else if (err.message) {
    msg = err.message;
  }
  showError(msg);
  setStatus("error", msg);
}
