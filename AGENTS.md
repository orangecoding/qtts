# Agent Guidelines for qtts

This document provides coding agents with essential information about the qtts (Qwen3-TTS CLI) project.

## Project Overview

qtts is a Python CLI tool for generating high-quality speech from text using Qwen3-TTS models locally. It supports voice cloning, preset voices, and voice design across 10+ languages.

**Tech Stack:** Python 3.12+, PyTorch, Click, Qwen3-TTS, soundfile, pydub

**Primary File:** `qtts.py` (336 lines) - Main CLI implementation

## Setup & Environment

### Prerequisites
- Python 3.12 (recommended) - versions 3.13/3.14 may have `llvmlite` build issues
- ffmpeg and sox (for MP3 conversion)
- CUDA (optional, for GPU acceleration)

### Environment Setup
```bash
# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: GPU acceleration (NVIDIA only)
pip install flash-attn --no-build-isolation

# Make executable
chmod +x qtts.py
```

### Running the CLI
```bash
# Basic usage
./qtts.py "Hello world!" -s Ryan -l English

# Voice cloning
./qtts.py "Text" -m clone --ref-audio voice.wav --ref-text "reference"

# Voice design
./qtts.py "Text" -m design -i "Male voice, 30s, dramatic tone"
```

## Testing & Development

### Manual Testing
```bash
# Test basic generation
./qtts.py "Hello world" -s Vivian -l English -o test_output.mp3

# Test all speakers
./qtts.py --list-speakers

# Test languages
./qtts.py --list-languages

# Test voice cloning (requires reference audio)
./qtts.py "Test" -m clone --ref-audio samples/test_ryan.mp3 \
  --ref-text "reference text" -o test_clone.mp3

# Test CPU mode
./qtts.py "Test" --device cpu -o test_cpu.mp3
```

### No Automated Tests
This project currently has no unit tests or test framework. Manual testing is required.

## Code Style & Conventions

### Python Style
- **Python Version:** 3.12+ (use 3.12 for best compatibility)
- **Line Length:** ~80-100 characters (not strictly enforced)
- **Indentation:** 4 spaces
- **Quotes:** Single quotes for strings, double quotes for docstrings

### Imports
Order imports as follows:
1. Standard library (os, sys, pathlib, typing)
2. Third-party packages (click, torch, soundfile, pydub)
3. Local modules (if any)

Example from qtts.py:
```python
import os
import sys
from pathlib import Path
from typing import Optional

import click
import torch
import soundfile as sf
from pydub import AudioSegment
```

### Type Hints
- Use type hints for function parameters and return values
- Import from `typing` module (Optional, str, etc.)
- Examples:
  ```python
  def load_model(model_name: str, device: str):
  def generate_clone_mode(...) -> tuple:
  model_path: Optional[str] = None
  ```

### Naming Conventions
- **Functions:** `snake_case` (e.g., `load_model`, `generate_clone_mode`)
- **Variables:** `snake_case` (e.g., `model_name`, `sample_rate`, `wav_data`)
- **Constants:** `UPPER_SNAKE_CASE` (though none currently defined)
- **Private/Internal:** Prefix with underscore (e.g., `_model_cache`)
- **CLI Options:** Kebab-case in Click decorators (`--ref-audio`, `--list-speakers`)

### Docstrings
Use simple docstrings for functions:
```python
def get_device():
    """Determine the best available device."""
    # implementation
```

For complex functions, describe parameters and behavior:
```python
def generate_clone_mode(
    text: str,
    ref_audio: str,
    ref_text: str,
    language: str,
    device: str,
    model_path: Optional[str] = None
):
    """Generate speech using voice cloning."""
    # implementation
```

### Error Handling
- Use try-except blocks for operations that can fail
- Print user-friendly error messages with `click.echo(..., err=True)`
- Exit with `sys.exit(1)` on critical errors
- Example:
  ```python
  try:
      wavs, sr = model.generate_voice_clone(...)
      return wavs[0], sr
  except Exception as e:
      click.echo(f"Error during generation: {e}", err=True)
      sys.exit(1)
  ```

### Click CLI Patterns
- Use `@click.command()` for main entry point
- Use `@click.argument()` for required positional args
- Use `@click.option()` for optional flags with defaults
- Provide helpful descriptions in `help=` parameter
- Use `is_flag=True` for boolean flags
- Use `click.Choice()` for enumerated options

### Code Organization
- Global variables: Prefix with underscore (`_model_cache`)
- Helper functions before main CLI function
- Main CLI function decorated with Click at bottom
- Entry point: `if __name__ == '__main__':`

### Device Handling
Auto-detect device in order of preference:
1. CUDA (nvidia GPU)
2. MPS (Apple Silicon)
3. CPU (fallback)

```python
def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
```

### Model Caching
Use global cache to avoid reloading models:
```python
_model_cache = {}

def load_model(model_name: str, device: str):
    cache_key = f"{model_name}_{device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    # ... load and cache model
```

## Architecture Notes

### Three Generation Modes
1. **Custom Mode** (default): Preset speakers with optional emotion control
   - Model: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
   - Speakers: Vivian, Ryan, Aiden, etc.

2. **Clone Mode**: Voice cloning from reference audio
   - Model: `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
   - Requires: reference audio file + transcript

3. **Design Mode**: Generate voice from text description
   - Model: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
   - Requires: voice description in `--instruct`

### Audio Pipeline
1. Generate WAV using qwen-tts model
2. Save temporary WAV with soundfile
3. Convert to MP3 using pydub (if output is .mp3)
4. Clean up temporary WAV file

## Common Pitfalls

1. **Python 3.13/3.14 Issues:** Use Python 3.12 to avoid llvmlite/numba build failures
2. **Missing ffmpeg:** MP3 conversion requires ffmpeg; fallback to WAV if unavailable
3. **CUDA out of memory:** Use `--device cpu` or smaller model variant
4. **Model downloads:** First run downloads 600MB-1.7GB models from HuggingFace
5. **Flash attention:** Optional; CLI gracefully falls back to eager attention

## File Structure
```
qtts/
├── qtts.py              # Main CLI application
├── requirements.txt     # Python dependencies
├── README.md           # User documentation
├── LICENSE             # Apache 2.0 license
├── samples/            # Sample audio outputs
└── venv/               # Virtual environment (gitignored)
```

## Making Changes

When modifying the code:
1. Preserve the existing code style and patterns
2. Test all three modes (custom, clone, design) if changing generation logic
3. Test on CPU and GPU if changing device handling
4. Ensure error messages are user-friendly
5. Update README.md if adding new features or options
6. Keep qtts.py as a single-file CLI for simplicity
