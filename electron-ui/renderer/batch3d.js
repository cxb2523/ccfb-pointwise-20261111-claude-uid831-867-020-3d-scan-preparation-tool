// Thin renderer controller for the 3D model batch service.
// Security boundary: this file never touches the filesystem or child
// processes. It only sends start/cancel requests through the preload bridge
// and renders the NDJSON events the main process forwards from the Python
// service child.

(() => {
  const bridge = window.scanprep || {
    chooseFolder: async () => null,
    startBatch3d: async () => ({ code: 1 }),
    cancelBatch3d: async () => ({ cancelled: false }),
    openPath: async () => ({ opened: false }),
    onBatch3dEvent: () => () => {}
  };

  const inputPathEl = document.querySelector("#batch3dInputPath");
  const chooseButton = document.querySelector("#batch3dChooseButton");
  const concurrencyEl = document.querySelector("#batch3dConcurrency");
  const decimateEl = document.querySelector("#batch3dDecimate");
  const decimateLabel = document.querySelector("#batch3dDecimateLabel");
  const textureFormatEl = document.querySelector("#batch3dTextureFormat");
  const runButton = document.querySelector("#batch3dRunButton");
  const openOutputButton = document.querySelector("#batch3dOpenOutputButton");
  const summaryEl = document.querySelector("#batch3dSummary");
  const progressBar = document.querySelector("#batch3dProgressBar");
  const taskListEl = document.querySelector("#batch3dTaskList");

  let isRunning = false;
  let totalTasks = 0;
  let finishedTasks = 0;
  const rows = new Map();
  let lastOutputDir = "";

  decimateEl?.addEventListener("input", () => {
    decimateLabel.textContent = `${decimateEl.value}%`;
  });

  chooseButton?.addEventListener("click", async () => {
    const folder = await bridge.chooseFolder({ title: "Choose model folder" });
    if (folder) inputPathEl.value = folder;
  });

  function baseName(pathText) {
    const parts = String(pathText).split(/[\\/]/).filter(Boolean);
    return parts[parts.length - 1] || pathText;
  }

  function siblingProcessed(inputPath) {
    // Display-only hint; the service independently computes and owns paths.
    if (!inputPath) return "";
    const trimmed = inputPath.replace(/[\\/]+$/, "");
    return `${trimmed}_processed`;
  }

  function ensureRow(taskIndex, source) {
    if (rows.has(taskIndex)) return rows.get(taskIndex);
    const row = document.createElement("div");
    row.className = "batch3d-task is-queued";
    row.innerHTML = `
      <div class="batch3d-task-head">
        <span class="batch3d-task-name"></span>
        <span class="batch3d-task-state">queued</span>
      </div>
      <div class="batch3d-task-detail quiet"></div>`;
    row.querySelector(".batch3d-task-name").textContent = baseName(source);
    taskListEl.querySelector(".quiet")?.remove();
    taskListEl.append(row);
    rows.set(taskIndex, row);
    return row;
  }

  function setRowState(row, state, detail) {
    row.classList.remove("is-queued", "is-running", "is-completed", "is-failed", "is-cancelled");
    row.classList.add(`is-${state}`);
    row.querySelector(".batch3d-task-state").textContent = state;
    if (detail !== undefined) row.querySelector(".batch3d-task-detail").textContent = detail;
  }

  function setOverallPercent(percent) {
    progressBar.style.width = `${Math.max(0, Math.min(100, percent))}%`;
  }

  function recomputeOverall(currentTaskPercent = 0) {
    if (!totalTasks) return;
    const partial = finishedTasks + (currentTaskPercent / 100);
    setOverallPercent((partial / totalTasks) * 100);
  }

  function setRunning(running) {
    isRunning = running;
    runButton.textContent = running ? "Stop Batch" : "Start 3D Batch";
    runButton.classList.toggle("stop-button", running);
    [inputPathEl, chooseButton, concurrencyEl, decimateEl, textureFormatEl]
      .forEach((el) => { if (el) el.disabled = running; });
  }

  function resetPanel() {
    rows.clear();
    finishedTasks = 0;
    totalTasks = 0;
    taskListEl.innerHTML = "";
    const hint = document.createElement("p");
    hint.className = "quiet";
    hint.textContent = "Queued models, per-file face counts and failure details appear here.";
    taskListEl.append(hint);
    setOverallPercent(0);
  }

  runButton?.addEventListener("click", async () => {
    if (isRunning) {
      await bridge.cancelBatch3d();
      summaryEl.textContent = "Cancelling after the current task stage...";
      return;
    }
    if (!inputPathEl.value) {
      summaryEl.textContent = "Choose a model folder or file first.";
      return;
    }

    resetPanel();
    lastOutputDir = siblingProcessed(inputPathEl.value);
    setRunning(true);
    summaryEl.textContent = "Starting service...";

    const config = {
      inputPath: inputPathEl.value,
      concurrency: Number(concurrencyEl.value),
      decimateRatio: Number(decimateEl.value) / 100,
      textureFormat: textureFormatEl.value,
      recursive: true,
      webpQuality: 90
    };

    // Resolves when the service child exits; events drive the live UI.
    const result = await bridge.startBatch3d(config);
    setRunning(false);
    if (summaryEl.textContent.startsWith("Starting")) {
      summaryEl.textContent = `Service exited (code ${result.code}).`;
    }
  });

  openOutputButton?.addEventListener("click", () => {
    if (lastOutputDir) bridge.openPath(lastOutputDir);
  });

  bridge.onBatch3dEvent((event) => {
    if (!event || !event.type) return;

    switch (event.type) {
      case "batch_started":
        totalTasks = Number(event.total) || 0;
        summaryEl.textContent =
          `Queued ${totalTasks} model(s) - ${event.concurrency} worker(s), ` +
          `${Math.round((event.decimate_ratio || 0) * 100)}% decimation, ` +
          `textures: ${event.texture_format}`;
        setOverallPercent(0);
        break;

      case "task_queued":
        ensureRow(event.task, event.source);
        break;

      case "task_started": {
        const row = ensureRow(event.task, event.source);
        setRowState(row, "running", "starting...");
        break;
      }

      case "task_progress": {
        const row = ensureRow(event.task, event.source);
        setRowState(row, "running", `${event.message || event.stage} (${event.percent}%)`);
        recomputeOverall(Number(event.percent) || 0);
        break;
      }

      case "textures_prepared": {
        const row = rows.get(event.task);
        if (!row) break;
        const textures = Array.isArray(event.textures) ? event.textures : [];
        const transcoded = textures.filter((t) => t.transcoded).length;
        const engines = [...new Set(textures.map((t) => t.engine))].join(", ");
        const existing = row.querySelector(".batch3d-task-detail").textContent;
        row.querySelector(".batch3d-task-detail").textContent =
          `${existing} | ${transcoded}/${textures.length} textures transcoded (${engines || "none"})`;
        if (Array.isArray(event.warnings) && event.warnings.length) {
          row.querySelector(".batch3d-task-detail").textContent +=
            ` | warnings: ${event.warnings.join("; ")}`;
        }
        break;
      }

      case "task_completed": {
        const row = ensureRow(event.task, event.source);
        finishedTasks += 1;
        setRowState(
          row,
          "completed",
          `faces ${event.input_faces} -> ${event.output_faces} ` +
          `(${event.materials} material group(s)) -> ${baseName(event.output || "")}`
        );
        if (event.output) lastOutputDir = event.output.replace(/[\\/][^\\/]+$/, "");
        summaryEl.textContent = `Completed ${finishedTasks}/${totalTasks}`;
        recomputeOverall(0);
        break;
      }

      case "task_failed": {
        const row = ensureRow(event.task, event.source);
        finishedTasks += 1;
        // Failure is isolated: the row is marked failed but the batch continues.
        setRowState(row, "failed", `isolated failure: ${event.error || "unknown error"}`);
        summaryEl.textContent = `Completed ${finishedTasks}/${totalTasks} (failures isolated)`;
        recomputeOverall(0);
        break;
      }

      case "task_cancelled": {
        const row = ensureRow(event.task, event.source);
        finishedTasks += 1;
        setRowState(row, "cancelled", event.error || "cancelled");
        recomputeOverall(0);
        break;
      }

      case "batch_completed":
        summaryEl.textContent =
          `Done: ${event.completed} completed, ${event.failed} failed, ` +
          `${event.cancelled} cancelled of ${event.total} model(s).`;
        setOverallPercent(100);
        break;

      case "batch_cancelled":
        summaryEl.textContent =
          `Cancelled: ${event.completed} completed, ${event.failed} failed, ` +
          `${event.cancelled} cancelled of ${event.total} model(s).`;
        break;

      case "status":
        if (isRunning) summaryEl.textContent = event.message || summaryEl.textContent;
        break;

      case "error":
        summaryEl.textContent = `Service error: ${event.message || "unknown"}`;
        break;

      default:
        break;
    }
  });
})();
