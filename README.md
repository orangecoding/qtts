# qtts - Qwen3-TTS Command Line Interface

A simple, powerful CLI for generating high-quality speech from text using Qwen3-TTS models locally.

## Features

- **Voice Cloning**: Clone any voice from a 3-second audio sample
- **Preset Voices**: 9 premium voices covering multiple languages and dialects
- **Voice Design**: Generate custom voices from natural language descriptions
- **Multi-language**: Supports 10 major languages (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian)
- **Emotion Control**: Fine-tune tone, emotion, and speaking style with text instructions
- **High Quality**: 12Hz tokenizer for natural-sounding speech
- **MP3 Export**: Direct MP3 output for easy sharing

## Installation

### Prerequisites

1. **Python 3.12** (recommended - see Troubleshooting if using 3.13/3.14)
2. **ffmpeg** (for MP3 conversion):
   ```bash
   # macOS
   brew install ffmpeg sox
   
   # Ubuntu/Debian
   sudo apt install ffmpeg sox
   
   # Windows (using chocolatey)
   choco install ffmpeg sox
   ```

3. **CUDA** (optional, for GPU acceleration - highly recommended)

### Setup

1. Clone or download this repository:
   ```bash
   cd /Users/daliusdobravolskas/projects/qwen3
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

4. (Optional) For better performance with NVIDIA GPU only:
   ```bash
   # Only works on Linux/Windows with CUDA-capable GPU
   # Will NOT work on macOS or CPU-only systems
   pip install flash-attn --no-build-isolation
   ```
   
   **Note:** Flash-attention requires CUDA GPU. If installation fails, you can skip this step - the CLI will work fine without it (just a bit slower).

5. Make the script executable:
   ```bash
   chmod +x qtts.py
   ```

6. (Optional) Add to PATH for global access:
   ```bash
   # Add this line to your ~/.bashrc or ~/.zshrc
   export PATH="/Users/daliusdobravolskas/projects/qwen3:$PATH"
   ```

## Quick Start

### Basic Usage (Preset Voice)

```bash
./qtts.py "Hello, welcome to Qwen3-TTS!" -s Vivian -l English
```

This generates `output.mp3` with the Vivian voice speaking in English.

### With Emotion Control

```bash
./qtts.py "I'm so excited to meet you!" -s Ryan -i "Very happy and energetic"
```

### Voice Cloning

Clone a voice from a reference audio:

```bash
./qtts.py "This is my cloned voice" -m clone \
  --ref-audio path/to/reference.wav \
  --ref-text "The text spoken in the reference audio"
```

### Voice Design

Create a unique voice from a description:

```bash
./qtts.py "Hello there!" -m design \
  -i "Young male voice, 25 years old, cheerful and confident tone"
```

## Usage Guide

### Command Structure

```bash
./qtts.py [TEXT] [OPTIONS]
```

### Core Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output` | Output file path | `output.mp3` |
| `-m, --mode` | Generation mode: `custom`, `clone`, or `design` | `custom` |
| `-l, --language` | Target language (see list below) | `Auto` |

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
./qtts.py --list-speakers
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
| `--device` | Device: `cuda:0`, `cpu`, `mps` (auto-detected) |
| `--list-languages` | Show all supported languages |

### Supported Languages

Auto, Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

Use `--list-languages` to display the full list.

## Examples

### 1. English Speech with American Voice

```bash
./qtts.py "Welcome to the future of text-to-speech" -s Aiden -l English
```

### 2. Chinese Speech with Emotion

```bash
./qtts.py "其实我真的有发现，我是一个特别善于观察别人情绪的人。" \
  -s Vivian -l Chinese -i "用特别愤怒的语气说"
```

### 3. Japanese Speech

```bash
./qtts.py "こんにちは、世界！" -s Ono_Anna -l Japanese
```

### 4. Clone Voice from URL

```bash
./qtts.py "I am solving the equation" -m clone \
  --ref-audio "https://example.com/voice.wav" \
  --ref-text "Original text from the audio" \
  -l English
```

### 5. Design a Character Voice

```bash
./qtts.py "Prepare for trouble, and make it double!" -m design \
  -i "Male villain voice, 30s, dramatic and theatrical with a hint of menace" \
  -l English -o villain.mp3
```

### 6. Multiple Outputs

Generate multiple files with different voices:

```bash
./qtts.py "Hello world" -s Vivian -o output1.mp3
./qtts.py "Hello world" -s Ryan -o output2.mp3
./qtts.py "Hello world" -s Aiden -o output3.mp3
```

## Tips & Best Practices

### Voice Cloning
- Use high-quality reference audio (clear, minimal background noise)
- Reference audio should be 3-10 seconds long
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
- First run downloads models (~1-3GB), be patient
- GPU (CUDA) recommended for faster generation
- Models are cached after first use
- Use shorter texts for faster processing

### Language Selection
- Set specific language when known (faster than Auto)
- Auto mode works well for mixed-language text
- Each speaker performs best in their native language

## Troubleshooting

### Python 3.14/3.13 Installation Issues (macOS)

**Problem:** If you're using Python 3.14 or 3.13, you may encounter build errors with `llvmlite` (required by `librosa` which is needed by `qwen-tts`).

**Solution:** Use Python 3.12 instead:

```bash
# If you have Python 3.12 installed via Homebrew
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

### "CUDA out of memory"
- Use CPU mode: `--device cpu`
- Close other applications using GPU
- Use smaller model (0.6B instead of 1.7B)

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
- Use GPU if available
- Install flash-attention: `pip install flash-attn --no-build-isolation`
- Use 0.6B model instead of 1.7B

## Model Information

### Default Models

- **Custom Mode**: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` (~600MB)
- **Clone Mode**: `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (~600MB)
- **Design Mode**: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` (~1.7GB)

Models are automatically downloaded from HuggingFace on first use and cached locally.

### Hardware Requirements

**Minimum:**
- CPU: Any modern processor
- RAM: 8GB
- Storage: 5GB free

**Recommended:**
- GPU: NVIDIA GPU with 4GB+ VRAM (CUDA support)
- RAM: 16GB
- Storage: 10GB free

## Technical Details

- **Tokenizer**: Qwen3-TTS-Tokenizer-12Hz
- **Architecture**: Discrete multi-codebook LM
- **Precision**: bfloat16 (GPU) / float32 (CPU)
- **Sample Rate**: 24kHz output
- **Latency**: ~2-5 seconds per sentence (GPU)

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
