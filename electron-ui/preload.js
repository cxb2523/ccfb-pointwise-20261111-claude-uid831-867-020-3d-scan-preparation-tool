const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("scanprep", {
  chooseFolder: (options) => ipcRenderer.invoke("choose-folder", options),
  chooseFile: (options) => ipcRenderer.invoke("choose-file", options),
  chooseFiles: (options) => ipcRenderer.invoke("choose-files", options),
  defaultOutputFor: (inputPath) => ipcRenderer.invoke("default-output-for", inputPath),
  getDiagnostics: (config) => ipcRenderer.invoke("get-diagnostics", config),
  testGpu: () => ipcRenderer.invoke("backend-test-gpu"),
  runConfig: (config) => ipcRenderer.invoke("backend-run-config", config),
  prepareDisplayPreview: (config) => ipcRenderer.invoke("backend-display-preview", config),
  prepareSourcePreview: (config) => ipcRenderer.invoke("backend-source-preview", config),
  previewSharpness: (config) => ipcRenderer.invoke("backend-sharpness-preview", config),
  previewMask: (config) => ipcRenderer.invoke("backend-mask-preview", config),
  previewContrast: (config) => ipcRenderer.invoke("backend-contrast-preview", config),
  cancelBackend: () => ipcRenderer.invoke("backend-cancel"),
  openPath: (targetPath) => ipcRenderer.invoke("open-path", targetPath),
  // 3D model batch service: the renderer can only start/cancel a batch and
  // subscribe to events. It has no direct filesystem or child-process access.
  startBatch3d: (config) => ipcRenderer.invoke("batch3d-start", config),
  cancelBatch3d: () => ipcRenderer.invoke("batch3d-cancel"),
  onBatch3dEvent: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on("batch3d-event", listener);
    return () => ipcRenderer.removeListener("batch3d-event", listener);
  },
  onBackendEvent: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on("backend-event", listener);
    return () => ipcRenderer.removeListener("backend-event", listener);
  }
});
