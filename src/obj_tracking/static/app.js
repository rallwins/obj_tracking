const byId = (id) => document.getElementById(id);

const state = {
  file: null,
  resultUrl: null,
  streaming: false,
  toastTimer: null,
};

const videoInput = byId("video-input");
const dropzone = byId("dropzone");
const selectedFile = byId("selected-file");
const uploadError = byId("upload-error");
const resultsEmpty = byId("results-empty");
const processingState = byId("processing-state");
const resultVideoWrap = byId("result-video-wrap");
const resultVideo = byId("result-video");
const resultsActions = byId("results-actions");
const toast = byId("toast");
const confidenceInput = byId("confidence");
const confidenceValue = byId("confidence-value");

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function showToast(message, isError = false) {
  window.clearTimeout(state.toastTimer);
  toast.textContent = message;
  toast.classList.toggle("error", isError);
  toast.classList.remove("hidden");
  state.toastTimer = window.setTimeout(() => toast.classList.add("hidden"), 4800);
}

function setFile(file) {
  if (!file) return;
  const allowedExtensions = /\.(mp4|webm|mov|avi|mkv|m4v)$/i;
  if (!allowedExtensions.test(file.name)) {
    uploadError.textContent = "Choose a supported video: MP4, WEBM, MOV, AVI, MKV, or M4V.";
    uploadError.classList.remove("hidden");
    return;
  }
  if (file.size > 500 * 1024 * 1024) {
    uploadError.textContent = "This video is larger than the 500 MB upload limit.";
    uploadError.classList.remove("hidden");
    return;
  }
  uploadError.classList.add("hidden");
  state.file = file;
  byId("file-name").textContent = file.name;
  byId("file-size").textContent = formatBytes(file.size);
  selectedFile.classList.remove("hidden");
  byId("start-tracking").classList.remove("hidden");
  dropzone.classList.add("has-file");
}

function clearResult() {
  resultVideo.pause();
  resultVideo.removeAttribute("src");
  resultVideo.load();
  if (state.resultUrl) URL.revokeObjectURL(state.resultUrl);
  state.resultUrl = null;
  byId("download-result").href = "#";
  resultVideoWrap.classList.add("hidden");
  resultsActions.classList.add("hidden");
  resultsEmpty.classList.remove("hidden");
  byId("result-meta").textContent = "";
}

function setConfidence(value) {
  const min = Number(confidenceInput.min);
  const max = Number(confidenceInput.max);
  const progress = ((Number(value) - min) / (max - min)) * 100;
  confidenceInput.style.background = `linear-gradient(to right, var(--green) 0%, var(--green) ${progress}%, #303a32 ${progress}%, #303a32 100%)`;
  confidenceValue.value = Number(value).toFixed(2);
  confidenceValue.textContent = Number(value).toFixed(2);
}

function selectMode(mode) {
  const uploadActive = mode === "upload";
  byId("upload-tab").classList.toggle("active", uploadActive);
  byId("webcam-tab").classList.toggle("active", !uploadActive);
  byId("upload-tab").setAttribute("aria-selected", String(uploadActive));
  byId("webcam-tab").setAttribute("aria-selected", String(!uploadActive));
  byId("upload-panel").classList.toggle("hidden", !uploadActive);
  byId("webcam-panel").classList.toggle("hidden", uploadActive);
  if (state.streaming) stopCamera();
}

async function checkServer() {
  const badge = byId("server-status");
  const label = byId("server-status-text");
  try {
    const response = await fetch("/health", { cache: "no-store" });
    if (!response.ok) throw new Error("Server did not respond");
    badge.classList.remove("checking", "offline");
    label.textContent = "Server online";
  } catch {
    badge.classList.remove("checking");
    badge.classList.add("offline");
    label.textContent = "Server unavailable";
  }
}

function readableApiError(body, status) {
  if (typeof body?.detail === "string") return body.detail;
  if (Array.isArray(body?.detail)) {
    return body.detail.map((item) => item.msg).filter(Boolean).join("; ");
  }
  return `The tracking request failed (HTTP ${status}).`;
}

async function processUpload() {
  if (!state.file) {
    showToast("Choose a video file before starting tracking.", true);
    videoInput.click();
    return;
  }

  const form = new FormData();
  form.append("video", state.file);
  form.append("confidence", confidenceInput.value);
  uploadError.classList.add("hidden");
  resultsEmpty.classList.add("hidden");
  resultVideoWrap.classList.add("hidden");
  resultsActions.classList.add("hidden");
  processingState.classList.remove("hidden");

  try {
    const response = await fetch("/track", { method: "POST", body: form });
    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      throw new Error(readableApiError(errorBody, response.status));
    }
    const videoBlob = await response.blob();
    clearResult();
    state.resultUrl = URL.createObjectURL(videoBlob);
    resultVideo.src = state.resultUrl;
    byId("download-result").href = state.resultUrl;
    resultVideoWrap.classList.remove("hidden");
    resultsActions.classList.remove("hidden");
    resultsEmpty.classList.add("hidden");
    byId("result-meta").textContent = `${state.file.name} · ${formatBytes(state.file.size)} · CONFIDENCE ${Number(confidenceInput.value).toFixed(2)}`;
    showToast("Tracking complete. Your annotated video is ready.");
  } catch (error) {
    resultsEmpty.classList.remove("hidden");
    uploadError.textContent = error.message || "The video could not be processed.";
    uploadError.classList.remove("hidden");
    showToast(error.message || "The video could not be processed.", true);
  } finally {
    processingState.classList.add("hidden");
  }
}

function startCamera() {
  const stream = byId("camera-stream");
  const cameraError = byId("camera-error");
  const source = byId("camera-index").value;
  const confidence = confidenceInput.value;
  cameraError.classList.add("hidden");
  stream.onload = () => {
    byId("camera-placeholder").classList.add("hidden");
    stream.classList.remove("hidden");
    byId("preview-live-label").classList.remove("hidden");
  };
  stream.onerror = () => {
    stopCamera();
    cameraError.textContent = "Could not connect to the camera. Check the server logs and make sure a camera is available on the server machine.";
    cameraError.classList.remove("hidden");
  };
  stream.src = `/stream?source=${encodeURIComponent(source)}&confidence=${encodeURIComponent(confidence)}&t=${Date.now()}`;
  state.streaming = true;
  byId("camera-button").innerHTML = '<span class="stop-square" aria-hidden="true"></span><span>Stop camera</span>';
  byId("camera-button").classList.add("button-stop");
  byId("camera-placeholder").classList.add("hidden");
  byId("preview-live-label").classList.remove("hidden");
  resultsEmpty.classList.add("hidden");
  resultVideoWrap.classList.add("hidden");
  resultsActions.classList.add("hidden");
  processingState.classList.add("hidden");
}

function stopCamera() {
  const stream = byId("camera-stream");
  stream.onerror = null;
  stream.onload = null;
  stream.removeAttribute("src");
  stream.classList.add("hidden");
  byId("camera-placeholder").classList.remove("hidden");
  byId("preview-live-label").classList.add("hidden");
  byId("camera-button").innerHTML = '<svg class="button-play" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="m5 3.5 7 4.5-7 4.5v-9Z" fill="currentColor"/></svg><span>Start camera</span>';
  byId("camera-button").classList.remove("button-stop");
  state.streaming = false;
  if (!resultVideoWrap.classList.contains("hidden")) return;
  resultsEmpty.classList.remove("hidden");
}

byId("upload-tab").addEventListener("click", () => selectMode("upload"));
byId("webcam-tab").addEventListener("click", () => selectMode("webcam"));
dropzone.addEventListener("click", () => {
  videoInput.click();
});
videoInput.addEventListener("change", () => setFile(videoInput.files?.[0]));
byId("remove-file").addEventListener("click", () => {
  state.file = null;
  videoInput.value = "";
  selectedFile.classList.add("hidden");
  byId("start-tracking").classList.add("hidden");
  dropzone.classList.remove("has-file");
  uploadError.classList.add("hidden");
});

for (const eventName of ["dragenter", "dragover"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
}
dropzone.addEventListener("drop", (event) => setFile(event.dataTransfer?.files?.[0]));
byId("start-tracking").addEventListener("click", processUpload);

confidenceInput.addEventListener("input", () => setConfidence(confidenceInput.value));
byId("reset-settings").addEventListener("click", () => {
  confidenceInput.value = "0.3";
  setConfidence(confidenceInput.value);
});
byId("camera-button").addEventListener("click", () => {
  if (state.streaming) stopCamera();
  else startCamera();
});
byId("clear-results").addEventListener("click", clearResult);
byId("download-result").addEventListener("click", (event) => {
  if (!state.resultUrl) event.preventDefault();
});
confidenceInput.addEventListener("change", () => {
  if (state.streaming) {
    stopCamera();
    startCamera();
  }
});
document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "u") {
    event.preventDefault();
    videoInput.click();
  }
});
window.addEventListener("beforeunload", () => {
  if (state.resultUrl) URL.revokeObjectURL(state.resultUrl);
});

byId("camera-index").addEventListener("change", () => {
  if (state.streaming) {
    stopCamera();
    startCamera();
  }
});

setConfidence(confidenceInput.value);
checkServer();
window.setInterval(checkServer, 30000);
