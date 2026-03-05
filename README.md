# qtts - Qwen3-TTS Command Line Interface

A simple, powerful CLI for generating high-quality speech from text using Qwen3-TTS models locally. CPU-only, production-ready, with a Node.js wrapper for programmatic integration.

## Features

- **Voice Cloning**: Clone any voice from a 3-second audio sample
- **Preset Voices**: 9 premium voices covering multiple languages and dialects
- **Voice Design**: Generate custom voices from natural language descriptions
- **Multi-language**: Supports 10 major languages (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian)
- **Emotion Control**: Fine-tune tone, emotion, and speaking style with text instructions
- **Speed Control**: Adjust playback speed from 50 % to 150 % of normal
- **High Quality**: 12Hz tokenizer for natural-sounding speech
- **MP3 Export**: Direct MP3 output for easy sharing
- **Node.js Wrapper**: ESM module with async API for server-side integration
- **Self-healing**: Automatic cleanup, garbage collection, and hard timeout protection

## Quick Start

```bash
git clone <repo-url>
cd qtts
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./qtts.py "Hello world!" -s Ryan -l English
```

## Installation

### Prerequisites

1. **Python 3.12** (recommended — see Troubleshooting if using 3.13/3.14)
2. **ffmpeg** (for MP3 conversion):
   ```bash
   # macOS
   brew install ffmpeg sox

   # Ubuntu/Debian
   sudo apt install ffmpeg sox

   # Windows (using chocolatey)
   choco install ffmpeg sox
   ```
3. **Node.js 18+** (only if using the Node.js wrapper)

### Setup

1. Clone or download this repository:
   ```bash
   cd qtts
   ```

2. Create a virtual environment with Python 3.12:
   ```bash
   # macOS/Linux with brew-installed python3.12
   python3.12 -m venv venv
   source venv/bin/activate

   # Or use conda
   conda create -n qwen3-tts python=3.12 -y
   conda activate qwen3-tts
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Make the script executable:
   ```bash
   chmod +x qtts.py
   ```

### System-Wide Installation (Optional)

To make `qtts` available from anywhere on your system, use the wrapper script method:

> **Note:** A `pyproject.toml` is included for future pip/PyPI distribution, but `pip install` is not recommended due to potential llvmlite build issues on Python 3.12+. The wrapper script method below is the recommended approach.

1. Make the wrapper script executable:
   ```bash
   chmod +x qtts-wrapper.sh
   ```

2. Create a symlink in your user bin directory (no sudo required):
   ```bash
   ln -sf "$(pwd)/qtts-wrapper.sh" ~/.local/bin/qtts
   ```

   **Note:** Make sure `~/.local/bin` is in your PATH. If not, add this to your `~/.bashrc` or `~/.zshrc`:
   ```bash
   export PATH="$HOME/.local/bin:$PATH"
   ```

3. Alternatively, for system-wide installation (requires sudo):
   ```bash
   sudo ln -sf "$(pwd)/qtts-wrapper.sh" /usr/local/bin/qtts
   ```

4. Verify installation:
   ```bash
   which qtts
   qtts --list-speakers
   ```

After installation, you can run `qtts` from any directory without the `./` prefix or activating the virtual environment.

## Usage

### Command Structure

```bash
# If installed system-wide:
qtts [TEXT] [OPTIONS]

# Or from the project directory:
./qtts.py [TEXT] [OPTIONS]
```

**Note:** The rest of this guide uses `qtts` for brevity. If not installed system-wide, use `./qtts.py` instead.

### Core Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output` | Output file path (.mp3 or .wav) | `output.mp3` |
| `-m, --mode` | Generation mode: `custom`, `clone`, or `design` | `custom` |
| `-l, --language` | Target language (see list below) | `Auto` |
| `--speed` | Speed in percent (100 = normal, <100 = slower, >100 = faster) | `100` |

### Mode-Specific Options

#### Custom Voice Mode (Preset Speakers)

| Option | Description | Default |
|--------|-------------|---------|
| `-s, --speaker` | Preset speaker name | `Vivian` |
| `-i, --instruct` | Emotion/tone instruction (optional) | None |

**Available Speakers:**

| Speaker | Description | Native Language |
|---------|-------------|-----------------|
| `Vivian` | Bright, slightly edgy young female | Chinese |
| `Serena` | Warm, gentle young female | Chinese |
| `Uncle_Fu` | Seasoned male, low and mellow | Chinese |
| `Dylan` | Youthful Beijing male, clear | Chinese (Beijing) |
| `Eric` | Lively Chengdu male, husky | Chinese (Sichuan) |
| `Ryan` | Dynamic male, strong rhythm | English |
| `Aiden` | Sunny American male | English |
| `Ono_Anna` | Playful Japanese female | Japanese |
| `Sohee` | Warm Korean female | Korean |

List all speakers:
```bash
qtts --list-speakers
```

#### Clone Mode (Voice Cloning)

| Option | Description | Required |
|--------|-------------|----------|
| `--ref-audio` | Path or URL to reference audio (3+ seconds) | Yes |
| `--ref-text` | Transcript of the reference audio | Yes |

#### Design Mode (Voice Description)

| Option | Description | Required |
|--------|-------------|----------|
| `-i, --instruct` | Natural language voice description | Yes |

### Advanced Options

| Option | Description |
|--------|-------------|
| `--model` | Custom model path or HuggingFace ID |
| `--list-speakers` | Show all available preset speakers |
| `--list-languages` | Show all supported languages |

### Supported Languages

Auto, Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

Use `--list-languages` to display the full list.

## Examples

### 1. English Speech with American Voice

```bash
qtts "Welcome to the future of text-to-speech" -s Aiden -l English
```

### 2. Chinese Speech with Emotion

```bash
qtts "其实我真的有发现，我是一个特别善于观察别人情绪的人。" \
  -s Vivian -l Chinese -i "用特别愤怒的语气说"
```

### 3. Japanese Speech

```bash
qtts "こんにちは、世界！" -s Ono_Anna -l Japanese
```

### 4. Clone Voice from URL

```bash
qtts "I am solving the equation" -m clone \
  --ref-audio "https://example.com/voice.wav" \
  --ref-text "Original text from the audio" \
  -l English
```

### 5. Design a Character Voice

```bash
qtts "Prepare for trouble, and make it double!" -m design \
  -i "Male villain voice, 30s, dramatic and theatrical with a hint of menace" \
  -l English -o villain.mp3
```

### 6. Adjust Speed

```bash
# Slower speech (80 % speed)
qtts "Take your time" -s Ryan --speed 80

# Faster speech (130 % speed)
qtts "Hurry up!" -s Aiden --speed 130
```

### 7. Multiple Outputs

```bash
qtts "Hello world" -s Vivian -o output1.mp3
qtts "Hello world" -s Ryan -o output2.mp3
qtts "Hello world" -s Aiden -o output3.mp3
```

## Node.js Wrapper

`qtts-wrapper.js` is an ESM module that wraps the Python CLI for use in Node.js applications. It provides an async `synthesize` function suitable for production server-side integration.

### Programmatic Usage

```js
import { synthesize } from "./qtts-wrapper.js";

try {
  const outputPath = await synthesize({
    text:    "Hello world",        // required — text to synthesise
    output:  "/tmp/out.mp3",       // required — output file path
    mode:    "custom",             // "clone" | "custom" | "design" (default: "custom")
    speed:   50,                   // 0–100 slider, 50 = normal (default: 50)
    model:   null,                 // custom model path or HuggingFace ID
    refText: null,                 // reference transcript (required for clone mode)
  });
  console.log("Audio saved to:", outputPath);
} catch (err) {
  // err.code: 1 = validation/runtime error, 2 = timeout
  console.error("Synthesis failed:", err.message, "code:", err.code);
}
```

### CLI Usage

```bash
node qtts-wrapper.js --text "Hello" --output out.mp3 --mode custom --speed 50
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | Yes | — | Text to synthesise |
| `output` | string | Yes | — | Output file path |
| `mode` | string | No | `"custom"` | `"clone"`, `"custom"`, or `"design"` |
| `speed` | number | No | `50` | 0–100 slider (see speed mapping below) |
| `model` | string | No | `null` | Custom model path or HuggingFace ID |
| `refText` | string | No | `null` | Reference transcript (required for clone mode) |

### Speed Mapping

The wrapper uses a 0–100 slider that maps linearly to the Python CLI's percent scale:

| Slider value | qtts `--speed` | Effect |
|-------------|----------------|--------|
| `0` | `50 %` | Half speed (slowest) |
| `50` | `100 %` | Normal speed |
| `100` | `150 %` | 1.5× speed (fastest) |

### Error Handling

The `synthesize` function returns a Promise that:

- **Resolves** with the output file path (string) on success
- **Rejects** with an `Error` whose `.code` property is the numeric exit code:
  - `1` — validation or runtime error
  - `2` — timeout (Python process killed after 30 minutes)

### Timeout & Safety

- The Python process is killed with `SIGKILL` after **30 minutes** if it hasn't completed
- If the Node.js process receives `SIGTERM` or `SIGINT`, the child Python process is killed automatically
- stdout/stderr from the Python process is forwarded to the Node.js process

## Architecture

```
┌──────────────────┐     import      ┌───────────────────┐
│  Your Node.js    │ ──────────────▶ │  qtts-wrapper.js  │
│  Application     │   synthesize()  │  (ESM module)     │
└──────────────────┘                 └────────┬──────────┘
                                              │ execFile
                                              ▼
                                     ┌───────────────────┐
                                     │  qtts-wrapper.sh  │
                                     │  (venv activator) │
                                     └────────┬──────────┘
                                              │ exec
                                              ▼
                                     ┌───────────────────┐
                                     │  qtts.py          │
                                     │  (CPU inference)  │
                                     └───────────────────┘
```

- **qtts-wrapper.js** — Node.js ESM wrapper. Validates input, maps speed, spawns the Python process with timeout protection, and exposes the async `synthesize` API.
- **qtts-wrapper.sh** — Bash helper that resolves its real path (follows symlinks), activates the Python virtual environment, and `exec`s `qtts.py` with all arguments forwarded.
- **qtts.py** — CPU-only Python CLI built on Click. Handles model loading (with caching), speech generation across all three modes, speed adjustment, and MP3 conversion. Protected by a 30-minute `SIGALRM` timeout with automatic garbage collection.

## Tips & Best Practices

### Voice Cloning
- Use high-quality reference audio (clear, minimal background noise)
- Reference audio should be 3–10 seconds long
- Provide accurate transcription for best results
- Reference audio quality directly affects output quality

### Custom Voices
- Use each speaker's native language for best quality
- Keep instructions clear and concise
- Examples: "Very happy", "Angry and frustrated", "Calm and soothing"

### Voice Design
- Be specific: age, gender, tone, emotional state
- Good: "25-year-old female, energetic teacher voice, warm and encouraging"
- Avoid: "Nice voice"

### Performance
- First run downloads models (~1–3 GB) — be patient
- Models are cached after first use
- Use shorter texts for faster processing
- CPU inference is slower than GPU; expect longer generation times

### Language Selection
- Set a specific language when known (faster than Auto)
- Auto mode works well for mixed-language text
- Each speaker performs best in their native language

## Troubleshooting

### Python 3.14/3.13 Installation Issues (macOS)

**Problem:** If you're using Python 3.14 or 3.13, you may encounter build errors with `llvmlite` (required by `librosa` which is needed by `qwen-tts`).

**Solution:** Use Python 3.12 instead:

```bash
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Why this happens:** The `llvmlite` package (required by `numba`, which is required by `librosa`) doesn't have prebuilt wheels for Python 3.13/3.14 yet, and building from source requires LLVM 20 specifically (macOS typically has LLVM 15 or 21).

**Alternative:** Use conda environment which provides prebuilt numba/llvmlite:
```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
conda install -c conda-forge numba librosa -y
pip install -r requirements.txt
```

### "ffmpeg not found"
- Install ffmpeg (see Prerequisites)
- Or save as WAV: `-o output.wav`

### "Model download fails"
- Check internet connection
- Models download from HuggingFace on first use
- Use `--model` with local path if you pre-downloaded

### Poor audio quality
- Check reference audio quality (for clone mode)
- Try different speaker/language combinations
- Ensure text is clean (no special formatting)

### Slow generation
- CPU inference is expected to be slower than GPU
- Use the 0.6B models (custom/clone) for faster results
- Use shorter input text for faster processing

### Node.js wrapper errors
- Ensure `qtts-wrapper.sh` is executable: `chmod +x qtts-wrapper.sh`
- Ensure the Python virtual environment exists at `./venv/`
- Check that Node.js 18+ is installed (`node --version`)
- Timeout errors (code 2) may indicate the model is still downloading on first run

## Model Information

### Default Models

- **Custom Mode**: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` (~600 MB)
- **Clone Mode**: `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (~600 MB)
- **Design Mode**: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` (~1.7 GB)

Models are automatically downloaded from HuggingFace on first use and cached locally.

### Hardware Requirements

- **CPU**: Any modern processor
- **RAM**: 8 GB minimum, 16 GB recommended
- **Storage**: 5–10 GB free (for models and venv)

## Technical Details

- **Runtime**: CPU-only (float32, eager attention)
- **Tokenizer**: Qwen3-TTS-Tokenizer-12Hz
- **Architecture**: Discrete multi-codebook LM
- **Sample Rate**: 24 kHz output
- **Timeout**: 30-minute hard limit (both Python SIGALRM and Node.js process kill)

## License

This project uses Qwen3-TTS models which are licensed under Apache 2.0.

## Credits

Built on top of [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Cloud.

Model page: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base

## Support

For issues with the Qwen3-TTS models themselves, refer to:
- GitHub: https://github.com/QwenLM/Qwen3-TTS
- HuggingFace: https://huggingface.co/Qwen

For CLI-specific issues, check the troubleshooting section above.

---

Happy voice synthesizing with qtts!
