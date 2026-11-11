#!/usr/bin node
/*
 * End-to-end test for the 3D model batch service.
 *
 * Drives scanprep3d_service.py through child_process exactly like
 * electron-ui/main.js does, then asserts:
 *   - progress/queued/started events for a multi-material OBJ, a GLB and
 *     two corrupt files
 *   - per-file output face counts
 *   - texture transcoding and rewritten MTL/glTF references (textures load)
 *   - corrupt files are isolated (task_failed) and later tasks still run
 *   - reruns never overwrite: duplicate outputs get a _1 suffix
 *
 * Exits non-zero on the first failed assertion. No external test framework.
 */

const { spawn } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const assert = require("node:assert/strict");

const rootDir = path.resolve(__dirname, "..", "..");
const pythonExe = process.env.SCANPREP_PYTHON
  || (process.platform === "win32" ? "python" : "python3");
const serviceScript = path.join(rootDir, "scanprep3d_service.py");
const fixtureScript = path.join(__dirname, "fixtures", "make_fixtures.py");
const verifyScript = path.join(__dirname, "verify_outputs.py");

const workDir = fs.mkdtempSync(path.join(os.tmpdir(), "scanprep-e2e-"));
const fixtureDir = path.join(workDir, "fixtures");
const configPath = path.join(workDir, "config.json");
const outputDir = path.join(fixtureDir, "multi_material_processed");

let passed = 0;
function check(label, condition) {
  assert.ok(condition, label);
  passed += 1;
  console.log(`  PASS  ${label}`);
}

function runPython(scriptPath, args, input) {
  return new Promise((resolve, reject) => {
    const child = spawn(pythonExe, [scriptPath, ...args], {
      cwd: rootDir,
      env: { ...process.env, PYTHONUTF8: "1", PYTHONIOENCODING: "utf-8" },
      windowsHide: true
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => { stdout += chunk.toString(); });
    child.stderr.on("data", (chunk) => { stderr += chunk.toString(); });
    if (input) child.stdin.end(input);
    child.on("error", reject);
    child.on("close", (code) => resolve({ code, stdout, stderr }));
  });
}

function runBatch(config) {
  fs.writeFileSync(configPath, JSON.stringify(config, null, 2), "utf-8");
  return new Promise((resolve, reject) => {
    const child = spawn(pythonExe, [serviceScript, configPath], {
      cwd: rootDir,
      env: { ...process.env, PYTHONUTF8: "1", PYTHONIOENCODING: "utf-8" },
      windowsHide: true
    });
    const events = [];
    let buffer = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      buffer += chunk.toString();
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        try { events.push(JSON.parse(line)); }
        catch (err) { reject(new Error(`non-JSON event line: ${line}\n${err.message}`)); }
      }
    });
    child.stderr.on("data", (chunk) => { stderr += chunk.toString(); });
    child.on("error", reject);
    child.on("close", (code) => resolve({ code, events, stderr }));
  });
}

function byType(events, type) {
  return events.filter((event) => event.type === type);
}

async function main() {
  console.log(`Working directory: ${workDir}`);

  // 1. Fixtures: multi-material OBJ (PNG+TGA+JPEG), textured GLB, 2 corrupt.
  console.log("\n[1] Generating fixtures");
  const fixtureRun = await runPython(fixtureScript, [fixtureDir]);
  assert.equal(fixtureRun.code, 0, `fixture generation failed: ${fixtureRun.stderr}`);
  const fixtureFiles = fs.readdirSync(path.join(fixtureDir, "multi_material"));
  for (const name of ["materials_grid.obj", "materials.mtl", "spheres.glb",
    "tex_a.png", "tex_b.tga", "tex_c.jpg", "aaa_corrupt.obj", "aaa_corrupt.glb"]) {
    check(`fixture present: ${name}`, fixtureFiles.includes(name));
  }

  // 2. Run the batch serially through the same child-process path as the UI.
  console.log("\n[2] Running batch service (serial, WebP)");
  const firstRun = await runBatch({
    input_path: path.join(fixtureDir, "multi_material"),
    concurrency: 1,
    decimate_ratio: 0.5,
    texture_format: "webp",
    webp_quality: 85
  });
  assert.equal(firstRun.stderr, "", `unexpected stderr: ${firstRun.stderr}`);
  const events = firstRun.events;

  const started = byType(events, "batch_started")[0];
  check("batch_started reports 4 queued models", started && started.total === 4);
  check("concurrency defaults/holds at 1", started.concurrency === 1);
  check("4 task_queued events", byType(events, "task_queued").length === 4);
  check("4 task_started events", byType(events, "task_started").length === 4);

  // Progress events must cover every pipeline stage for the healthy files.
  const progressStages = new Set(
    byType(events, "task_progress").map((event) => event.stage));
  for (const stage of ["parse", "decimate", "textures", "write"]) {
    check(`task_progress covers stage: ${stage}`, progressStages.has(stage));
  }

  // 3. Corrupt files fail in isolation and the queue keeps going.
  console.log("\n[3] Failure isolation");
  const failed = byType(events, "task_failed");
  check("exactly 2 task_failed events", failed.length === 2);
  const failedNames = failed.map((event) =>
    path.basename(event.source)).sort();
  check("both corrupt files failed",
    JSON.stringify(failedNames) === JSON.stringify(["aaa_corrupt.glb", "aaa_corrupt.obj"]));
  check("failures carry error detail",
    failed.every((event) => event.error && event.error_type && event.traceback));

  // The GLB (task 3) starts AFTER the corrupt OBJ (task 1) failed:
  // failure isolation means later tasks continue.
  const firstFailIndex = events.findIndex((e) => e.type === "task_failed");
  const laterStartIndex = events.findIndex((e, i) =>
    i > firstFailIndex && e.type === "task_started");
  check("a later task starts after a failure (batch continues)", laterStartIndex > -1);

  // 4. Completed tasks report face counts; textures actually load.
  console.log("\n[4] Outputs: face counts and textures");
  const completed = byType(events, "task_completed");
  check("exactly 2 task_completed events", completed.length === 2);
  const objResult = completed.find((e) => e.source.endsWith(".obj"));
  const glbResult = completed.find((e) => e.source.endsWith(".glb"));
  check("OBJ decimated 1734 -> 864 faces",
    objResult.input_faces === 1734 && objResult.output_faces === 864);
  check("OBJ kept 3 material groups", objResult.materials === 3);
  check("GLB decimated 2560 -> 1280 faces",
    glbResult.input_faces === 2560 && glbResult.output_faces === 1280);
  check("GLB kept 2 materials", glbResult.materials === 2);

  const textureEvent = byType(events, "textures_prepared")
    .find((e) => e.source.endsWith(".obj"));
  const engines = textureEvent.textures.map((t) => t.engine).sort();
  check("PNG/TGA transcoded, JPEG passthrough",
    engines.join(",") === "passthrough,pillow-webp,pillow-webp");

  // Deep, independent verification on disk.
  const verify = await runPython(verifyScript, [outputDir]);
  if (verify.code !== 0) {
    console.log(verify.stdout);
    console.log(verify.stderr);
  }
  check("verify_outputs.py passes (groups, refs, decodable textures, GLB)",
    verify.code === 0);

  // Reference integrity directly on the written artifacts.
  const mtlText = fs.readFileSync(path.join(outputDir, "materials.mtl"), "utf-8");
  check("MTL rewrites PNG -> textures/tex_a.webp",
    /map_kd\s+textures\/tex_a\.webp/.test(mtlText));
  check("MTL rewrites TGA -> textures/tex_b.webp",
    /map_kd\s+textures\/tex_b\.webp/.test(mtlText));
  check("MTL keeps JPEG passthrough ref",
    /map_kd\s+textures\/tex_c\.jpg/.test(mtlText));
  const webpHeader = fs.readFileSync(
    path.join(outputDir, "textures", "tex_a.webp")).subarray(0, 4);
  check("tex_a.webp has RIFF header and loads",
    webpHeader.toString("ascii") === "RIFF");
  const glbBytes = fs.readFileSync(path.join(outputDir, "spheres.glb"));
  check("GLB declares EXT_texture_webp",
    glbBytes.includes(Buffer.from("EXT_texture_webp")));

  const batchDone = byType(events, "batch_completed")[0];
  check("batch_completed: 2 completed / 2 failed / 0 cancelled",
    batchDone.completed === 2 && batchDone.failed === 2
    && batchDone.cancelled === 0 && batchDone.total === 4);
  check("service exits 0 even with isolated per-file failures",
    firstRun.code === 0);

  // 5. Rerun: never overwrite; duplicates get a numeric suffix.
  console.log("\n[5] No-overwrite on rerun");
  const secondRun = await runBatch({
    input_path: path.join(fixtureDir, "multi_material"),
    concurrency: 1,
    decimate_ratio: 0.5,
    texture_format: "webp"
  });
  assert.equal(secondRun.code, 0, `rerun failed: ${secondRun.stderr}`);
  const rerunOutputs = fs.readdirSync(outputDir);
  check("rerun adds suffixed OBJ (materials_grid_1.obj)",
    rerunOutputs.includes("materials_grid_1.obj"));
  check("rerun adds suffixed GLB (spheres_1.glb)",
    rerunOutputs.includes("spheres_1.glb"));
  check("original first-run OBJ still intact",
    rerunOutputs.includes("materials_grid.obj"));

  // 6. Concurrency clamp: asking for 5 must settle at the cap of 2. Also
  // exercises texture_format "none" (keep originals) end-to-end.
  console.log("\n[6] Concurrency clamp + texture_format=none");
  const cappedRun = await runBatch({
    input_path: path.join(fixtureDir, "multi_material"),
    concurrency: 5,
    decimate_ratio: 0.5,
    texture_format: "none"
  });
  const cappedStart = byType(cappedRun.events, "batch_started")[0];
  check("requested concurrency 5 clamps to hard cap 2",
    cappedStart.concurrency === 2);
  const cappedDone = byType(cappedRun.events, "batch_completed")[0];
  check("texture_format=none still completes 2 and isolates 2 failures",
    cappedDone.completed === 2 && cappedDone.failed === 2);
  const noneCompleted = byType(cappedRun.events, "task_completed");
  check("none-run keeps original texture names on the OBJ output",
    noneCompleted.some((e) => e.textures
      && e.textures.every((t) => t.transcoded === false)));

  console.log(`\nAll ${passed} checks passed.`);
  console.log(`Artifacts retained for inspection at: ${workDir}`);
}

main().catch((error) => {
  console.error("\nE2E FAILED:");
  console.error(error);
  console.error(`\nArtifacts retained at: ${workDir}`);
  process.exit(1);
});
