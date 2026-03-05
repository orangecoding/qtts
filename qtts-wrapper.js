#!/usr/bin/env node

/**
 * qtts-wrapper.js — Production Node.js wrapper for the qtts Python TTS CLI.
 *
 * Exports an async `synthesize` function for programmatic use and doubles as
 * a CLI when executed directly.
 *
 * Programmatic usage:
 *   import { synthesize } from "./qtts-wrapper.js";
 *   const outputPath = await synthesize({
 *     text:    "Hello world",          // required
 *     output:  "/tmp/out.mp3",         // required
 *     mode:    "custom",               // "clone" | "custom" | "design" (default: "custom")
 *     model:   null,                   // custom model path or HuggingFace ID
 *     refText: null,                   // reference transcript (required for clone mode)
 *     speed:   50,                     // 0–100, where 50 = normal speed
 *   });
 *
 * CLI usage:
 *   node qtts-wrapper.js --text "Hello" --output out.mp3 [--mode custom] [--speed 50] ...
 *
 * Speed mapping (0–100 → percent passed to qtts --speed):
 *   0   → 50 %  (half speed — slowest)
 *   50  → 100 % (normal speed)
 *   100 → 150 % (1.5× speed — fastest)
 *
 * Error codes (rejected value / CLI exit code):
 *   1 — validation / runtime error
 *   2 — timeout (process killed)
 */

import { execFile } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// ---------- configuration ----------

/** Directory of this script (ESM equivalent of __dirname). */
const __dirname = dirname(fileURLToPath(import.meta.url));

/** Maximum milliseconds the Python process may run before being killed. */
const PROCESS_TIMEOUT_MS = 45 * 60 * 1000; // 30 minutes

/** Resolve qtts command — prefer the shell wrapper next to this script. */
const QTTS_CMD = join(__dirname, "qtts-wrapper.sh");

// ---------- helpers ----------

/**
 * Convert a 0–100 speed slider value to the percent value qtts expects.
 *
 * Mapping:  0 → 50,  50 → 100,  100 → 150  (linear interpolation)
 * This gives a usable range from half-speed to 1.5× speed.
 *
 * @param {number} sliderValue  Integer 0–100
 * @returns {number}            Integer percent for qtts --speed
 */
function mapSpeed(sliderValue) {
  return Math.round(50 + sliderValue);
}

/**
 * Validate parameters and return a normalised config object.
 * Throws on validation failure with a descriptive message.
 *
 * @param {object} params
 * @returns {object} Normalised config
 */
function validate(params) {
  if (!params.text) {
    throw new Error("--text is required.");
  }
  if (!params.output) {
    throw new Error("--output is required.");
  }

  // Speed: default to 50 (= normal), must be 0–100
  const rawSpeed = params.speed !== undefined ? Number(params.speed) : 50;
  if (Number.isNaN(rawSpeed) || rawSpeed < 0 || rawSpeed > 100) {
    throw new Error("--speed must be a number between 0 and 100.");
  }

  // Mode: default to "custom"
  const mode = (params.mode || "custom").toLowerCase();
  const validModes = ["clone", "custom", "design"];
  if (!validModes.includes(mode)) {
    throw new Error(`--mode must be one of: ${validModes.join(", ")}`);
  }

  return {
    model: params.model || null,
    text: params.text,
    mode,
    refText: params.refText || params["ref-text"] || null,
    speed: mapSpeed(rawSpeed),
    output: params.output,
  };
}

// ---------- core ----------

/**
 * Synthesise speech via the qtts Python CLI.
 *
 * @param {object} params
 * @param {string}  params.text     Text to synthesise (required)
 * @param {string}  params.output   Output file path (required)
 * @param {string}  [params.mode]   "clone" | "custom" | "design" (default: "custom")
 * @param {string}  [params.model]  Custom model path or HuggingFace ID
 * @param {string}  [params.refText] Reference transcript (clone mode)
 * @param {number}  [params.speed]  0–100 slider, 50 = normal (default: 50)
 *
 * @returns {Promise<string>} Resolves with the output file path on success.
 *                            Rejects with an error whose `.code` is the numeric exit code
 *                            (1 = runtime error, 2 = timeout).
 */
export function synthesize(params) {
  return new Promise((resolve, reject) => {
    // Validate — throws synchronously on bad input (code 1)
    let config;
    try {
      config = validate(params);
    } catch (validationErr) {
      const err = new Error(validationErr.message);
      err.code = 1;
      return reject(err);
    }

    // Build argument list for the qtts Python CLI
    const args = [
      config.text,                      // positional TEXT argument
      "-m", config.mode,
      "--speed", String(config.speed),
      "-o", config.output,
    ];

    if (config.model) {
      args.push("--model", config.model);
    }
    if (config.refText) {
      args.push("--ref-text", config.refText);
    }

    // Spawn the Python process with a hard timeout
    const child = execFile(QTTS_CMD, args, {
      timeout: PROCESS_TIMEOUT_MS,
      killSignal: "SIGKILL",
      maxBuffer: 10 * 1024 * 1024,     // 10 MB stdout/stderr buffer
      env: { ...process.env },
    }, (error, stdout, stderr) => {
      // Forward output so callers can capture logs if needed
      if (stdout) process.stdout.write(stdout);
      if (stderr) process.stderr.write(stderr);

      if (error) {
        const err = new Error(
          error.killed
            ? "Process killed (timeout or signal)."
            : error.message
        );
        err.code = error.killed ? 2 : (error.code || 1);
        return reject(err);
      }

      resolve(config.output);
    });

    // Safety net: if the Node process receives SIGTERM/SIGINT, kill the child
    const cleanup = () => {
      try { child.kill("SIGKILL"); } catch (_) { /* already dead */ }
    };
    process.on("SIGTERM", cleanup);
    process.on("SIGINT", cleanup);
  });
}

// ---------- CLI entry point ----------

/**
 * Parse CLI arguments into a plain object.
 * Supports `--key value` pairs; all values are strings.
 * @returns {Record<string, string>}
 */
function parseArgs() {
  const args = process.argv.slice(2);
  const params = {};
  for (let i = 0; i < args.length; i++) {
    if (args[i].startsWith("--") && i + 1 < args.length && !args[i + 1].startsWith("--")) {
      params[args[i].slice(2)] = args[i + 1];
      i++;
    }
  }
  return params;
}

/**
 * CLI entry — parses argv, calls synthesize, and exits with the appropriate code.
 */
async function main() {
  const params = parseArgs();

  console.log(`[qtts-wrapper] timeout : ${PROCESS_TIMEOUT_MS / 1000}s`);

  try {
    const outputPath = await synthesize(params);
    console.log(`[qtts-wrapper] done — ${outputPath}`);
    process.exit(0);
  } catch (err) {
    console.error(`[qtts-wrapper] Error: ${err.message}`);
    process.exit(typeof err.code === "number" ? err.code : 1);
  }
}

// Run CLI only when executed directly (not when imported as a module)
const isDirectRun = process.argv[1] &&
  fileURLToPath(import.meta.url) === process.argv[1];

if (isDirectRun) {
  main();
}
