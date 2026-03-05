#!/usr/bin/env python3
"""
qtts - Qwen3-TTS Command Line Interface (CPU-only)

Minimal, production-grade CLI for generating speech using Qwen3-TTS models on CPU.
Supports voice cloning, preset voices, and voice design modes.
"""

import gc
import os
import sys
import signal
from pathlib import Path
from typing import Optional

import click
import torch
import soundfile as sf
from pydub import AudioSegment


# Global model cache — avoids redundant reloads across calls in the same process
_model_cache = {}

# Hard timeout (seconds) for the entire generation process to prevent hangs
GENERATION_TIMEOUT = 2700


def _timeout_handler(signum, frame):
    """Signal handler that raises on SIGALRM to kill stuck generation."""
    raise TimeoutError("Generation exceeded timeout — aborting.")


def load_model(model_name: str):
    """
    Load a Qwen3-TTS model onto CPU with caching.
    Uses float32 and eager attention — the only viable options for CPU inference.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    click.echo(f"Loading model: {model_name} (CPU, float32)...")

    try:
        from qwen_tts import Qwen3TTSModel

        model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map="cpu",
            dtype=torch.float32,
            attn_implementation="eager",
        )
        _model_cache[model_name] = model
        click.echo("Model loaded successfully.")
        return model

    except ImportError:
        click.echo("Error: qwen_tts package not installed.", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Error loading model: {exc}", err=True)
        # Free any partial allocation before exiting
        gc.collect()
        sys.exit(1)


def generate_speech(mode, text, language, model_path, ref_audio=None, ref_text=None,
                    speaker="Vivian", instruct=None):
    """
    Unified generation entry point for all three modes.
    Returns (wav_data, sample_rate) on success; exits on failure.
    """
    # Resolve default model per mode
    defaults = {
        "clone":  "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "custom": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    }
    model_name = model_path or defaults[mode]
    model = load_model(model_name)

    try:
        if mode == "clone":
            wavs, sr = model.generate_voice_clone(
                text=text, language=language,
                ref_audio=ref_audio, ref_text=ref_text,
            )
        elif mode == "custom":
            wavs, sr = model.generate_custom_voice(
                text=text, language=language,
                speaker=speaker, instruct=instruct or "",
            )
        else:  # design
            wavs, sr = model.generate_voice_design(
                text=text, language=language,
                instruct=instruct,
            )
        return wavs[0], sr

    except Exception as exc:
        click.echo(f"Error during generation: {exc}", err=True)
        gc.collect()
        sys.exit(1)


def apply_speed(in_wav: str, out_wav: str, speed_percent: int):
    """
    Adjust playback speed via frame-rate manipulation.
    speed_percent: 100 = unchanged, <100 = slower, >100 = faster.
    Approximately preserves pitch by restoring the original frame rate.
    """
    # No-op shortcut — just move the file
    if speed_percent == 100:
        if in_wav != out_wav:
            os.replace(in_wav, out_wav)
        return

    if speed_percent <= 0:
        raise ValueError("speed_percent must be > 0")

    factor = speed_percent / 100.0
    sound = AudioSegment.from_wav(in_wav)

    # Alter speed by changing the frame rate, then restore to preserve pitch
    altered = sound._spawn(sound.raw_data, overrides={
        "frame_rate": int(sound.frame_rate * factor)
    })
    altered.set_frame_rate(sound.frame_rate).export(out_wav, format="wav")


def save_as_mp3(wav_path: str, mp3_path: str):
    """Convert a WAV file to 192 kbps MP3 and remove the source WAV."""
    try:
        AudioSegment.from_wav(wav_path).export(mp3_path, format="mp3", bitrate="192k")
        os.remove(wav_path)
    except Exception as exc:
        click.echo(f"Warning: MP3 conversion failed ({exc}). WAV kept at: {wav_path}", err=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.argument("text", type=str, required=False)
@click.option("-o", "--output", default="output.mp3", help="Output file path (default: output.mp3)")
@click.option("-m", "--mode", type=click.Choice(["clone", "custom", "design"], case_sensitive=False),
              default="custom", help="Generation mode")
@click.option("-l", "--language", default="Auto", help="Target language")
@click.option("--ref-audio", help="Reference audio for clone mode")
@click.option("--ref-text", help="Transcript of reference audio for clone mode")
@click.option("-s", "--speaker", default="Vivian", help="Preset speaker for custom mode")
@click.option("-i", "--instruct", help="Tone/emotion instruction or voice-design description")
@click.option("--model", help="Custom model path or HuggingFace ID")
@click.option("--speed", default=100, type=int, help="Speed in percent (100 = normal)")
@click.option("--list-speakers", is_flag=True, help="List available preset speakers")
@click.option("--list-languages", is_flag=True, help="List supported languages")
def main(text, output, mode, language, ref_audio, ref_text, speaker, instruct,
         model, speed, list_speakers, list_languages):
    """
    qtts — Generate speech from text using Qwen3-TTS (CPU-only).

    \b
    Examples:
      qtts "Hello world" -s Vivian -l English
      qtts "Hello world" --speed 90
      qtts "Hello" -m clone --ref-audio voice.wav --ref-text "ref text"
      qtts "Hello" -m design -i "Young male, cheerful"
    """

    # -- Informational flags (no generation needed) --
    if list_speakers:
        click.echo("Available speakers: Vivian, Serena, Uncle_Fu, Dylan, "
                    "Eric, Ryan, Aiden, Ono_Anna, Sohee")
        return

    if list_languages:
        click.echo("Supported languages: Auto, Chinese, English, Japanese, Korean, "
                    "German, French, Russian, Portuguese, Spanish, Italian")
        return

    # -- Input validation --
    if not text:
        click.echo("Error: TEXT argument is required.", err=True)
        sys.exit(1)

    if mode == "clone" and (not ref_audio or not ref_text):
        click.echo("Error: clone mode requires --ref-audio and --ref-text.", err=True)
        sys.exit(1)

    if mode == "design" and not instruct:
        click.echo("Error: design mode requires --instruct.", err=True)
        sys.exit(1)

    if speed <= 0:
        click.echo("Error: --speed must be > 0.", err=True)
        sys.exit(1)

    click.echo(f"=== qtts | mode={mode} | speed={speed}% | device=cpu ===")

    # -- Set a hard timeout (POSIX only) to kill stuck inference --
    prev_handler = None
    try:
        prev_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(GENERATION_TIMEOUT)
    except (AttributeError, OSError):
        pass  # SIGALRM not available on this platform — skip

    try:
        # Generate audio
        wav_data, sample_rate = generate_speech(
            mode, text, language, model,
            ref_audio=ref_audio, ref_text=ref_text,
            speaker=speaker, instruct=instruct,
        )

        # Write raw WAV
        output_path = Path(output)
        tmp_wav = output_path.with_suffix(".tmp.wav")
        sf.write(str(tmp_wav), wav_data, sample_rate)

        # Apply speed adjustment
        processed_wav = output_path.with_suffix(".speed.wav")
        try:
            apply_speed(str(tmp_wav), str(processed_wav), speed)
        finally:
            # Always clean up the raw temp file
            if tmp_wav.exists():
                try:
                    os.remove(tmp_wav)
                except OSError:
                    pass

        # Final output: convert to MP3 or keep WAV
        if output_path.suffix.lower() == ".mp3":
            save_as_mp3(str(processed_wav), str(output_path))
        else:
            if processed_wav != output_path:
                os.replace(str(processed_wav), str(output_path))

        click.echo(f"✓ Audio saved to: {output_path}")

    except TimeoutError:
        click.echo("Error: generation timed out.", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Unexpected error: {exc}", err=True)
        sys.exit(1)
    finally:
        # Cancel alarm and restore previous handler
        try:
            signal.alarm(0)
            if prev_handler is not None:
                signal.signal(signal.SIGALRM, prev_handler)
        except (AttributeError, OSError):
            pass
        gc.collect()


if __name__ == "__main__":
    main()
