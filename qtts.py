#!/usr/bin/env python3
"""
qtts - Qwen3-TTS Command Line Interface

A simple CLI for generating speech using Qwen3-TTS models locally.
Supports voice cloning, preset voices, and voice design.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import click
import torch
import soundfile as sf
from pydub import AudioSegment


# Model cache to avoid reloading
_model_cache = {}


def get_device():
    """Determine the best available device."""
    if torch.cuda.is_available():
        return 'cuda:0'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def load_model(model_name: str, device: str):
    """Load a Qwen3-TTS model with caching."""
    cache_key = f'{model_name}_{device}'

    if cache_key in _model_cache:
        return _model_cache[cache_key]

    click.echo(f'Loading model: {model_name}...')
    click.echo(f'Device: {device}')

    try:
        from qwen_tts import Qwen3TTSModel

        # Determine dtype based on device
        if device == 'cpu':
            dtype = torch.float32
            attn_impl = 'eager'
        else:
            dtype = torch.bfloat16
            # Try flash attention, fallback to eager if not available
            try:
                attn_impl = 'flash_attention_2'
            except Exception:
                attn_impl = 'eager'

        model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )

        _model_cache[cache_key] = model
        click.echo('Model loaded successfully!')
        return model

    except Exception as e:
        click.echo(f'Error loading model: {e}', err=True)
        sys.exit(1)


def save_as_mp3(wav_path: str, mp3_path: str):
    """Convert WAV to MP3."""
    try:
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format='mp3', bitrate='192k')
        # Clean up temporary WAV file
        os.remove(wav_path)
    except Exception as e:
        click.echo(f'Warning: Could not convert to MP3: {e}', err=True)
        click.echo(f'WAV file saved at: {wav_path}')


def apply_speed_percent(in_wav: str, out_wav: str, speed_percent: int):
    """
    Apply playback speed as percent.
    100 = unchanged
    90  = 10% slower
    110 = 10% faster

    This uses a simple pydub-based transform that approximately preserves pitch.
    """
    if speed_percent == 100:
        if in_wav != out_wav:
            os.replace(in_wav, out_wav)
        return

    if speed_percent <= 0:
        raise ValueError('speed percent must be > 0')

    speed = speed_percent / 100.0

    sound = AudioSegment.from_wav(in_wav)

    # Change frame rate to alter speed
    altered = sound._spawn(sound.raw_data, overrides={
        'frame_rate': int(sound.frame_rate * speed)
    })

    # Set back to original frame rate to approximately preserve pitch
    final = altered.set_frame_rate(sound.frame_rate)

    final.export(out_wav, format='wav')


def generate_clone_mode(
    text: str,
    ref_audio: str,
    ref_text: str,
    language: str,
    device: str,
    model_path: Optional[str] = None
):
    """Generate speech using voice cloning."""
    model_name = model_path or 'Qwen/Qwen3-TTS-12Hz-0.6B-Base'
    model = load_model(model_name, device)

    click.echo('Generating speech with voice cloning...')
    click.echo(f'Reference audio: {ref_audio}')

    try:
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        return wavs[0], sr
    except Exception as e:
        click.echo(f'Error during generation: {e}', err=True)
        sys.exit(1)


def generate_custom_mode(
    text: str,
    speaker: str,
    language: str,
    instruct: Optional[str],
    device: str,
    model_path: Optional[str] = None
):
    """Generate speech using preset custom voices."""
    model_name = model_path or 'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice'
    model = load_model(model_name, device)

    click.echo(f'Generating speech with custom voice: {speaker}')

    try:
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct or '',
        )
        return wavs[0], sr
    except Exception as e:
        click.echo(f'Error during generation: {e}', err=True)
        sys.exit(1)


def generate_design_mode(
    text: str,
    instruct: str,
    language: str,
    device: str,
    model_path: Optional[str] = None
):
    """Generate speech using voice design from description."""
    model_name = model_path or 'Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
    model = load_model(model_name, device)

    click.echo('Generating speech with voice design...')
    click.echo(f'Voice description: {instruct}')

    try:
        wavs, sr = model.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct,
        )
        return wavs[0], sr
    except Exception as e:
        click.echo(f'Error during generation: {e}', err=True)
        sys.exit(1)


@click.command()
@click.argument('text', type=str, required=False)
@click.option(
    '-o', '--output',
    default='output.mp3',
    help='Output MP3 file path (default: output.mp3)'
)
@click.option(
    '-m', '--mode',
    type=click.Choice(['clone', 'custom', 'design'], case_sensitive=False),
    default='custom',
    help='Generation mode: clone (voice cloning), custom (preset voices), or design (voice description)'
)
@click.option(
    '-l', '--language',
    default='Auto',
    help='Target language (Auto, Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian)'
)
@click.option(
    '--ref-audio',
    help='Reference audio file/URL for voice cloning (required for clone mode)'
)
@click.option(
    '--ref-text',
    help='Transcript of reference audio (required for clone mode)'
)
@click.option(
    '-s', '--speaker',
    default='Vivian',
    help='Preset speaker name for custom mode (Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee)'
)
@click.option(
    '-i', '--instruct',
    help='Instruction for tone/emotion control or voice design description'
)
@click.option(
    '--model',
    help='Custom model path or HuggingFace model ID'
)
@click.option(
    '--device',
    help='Device to use (cuda:0, cpu, mps). Auto-detected if not specified.'
)
@click.option(
    '--speed',
    default=100,
    type=int,
    help='Playback speed in percent (100 = normal, 90 = 10% slower, 110 = 10% faster)'
)
@click.option(
    '--list-speakers',
    is_flag=True,
    help='List available preset speakers'
)
@click.option(
    '--list-languages',
    is_flag=True,
    help='List supported languages'
)
def main(
    text: str,
    output: str,
    mode: str,
    language: str,
    ref_audio: Optional[str],
    ref_text: Optional[str],
    speaker: str,
    instruct: Optional[str],
    model: Optional[str],
    device: Optional[str],
    speed: int,
    list_speakers: bool,
    list_languages: bool
):
    """
    qtts - Generate speech from text using Qwen3-TTS

    Examples:

    \b
    # Use preset voice (default mode)
    qtts "Hello world" -s Vivian -l English

    \b
    # Make it 10% slower
    qtts "Hello world" --speed 90

    \b
    # Add emotion control
    qtts "I'm so excited!" -s Ryan -i "Very happy and energetic"

    \b
    # Voice cloning
    qtts "Hello world" -m clone --ref-audio voice.wav --ref-text "reference text"

    \b
    # Voice design
    qtts "Hello" -m design -i "Young male voice, cheerful tone"
    """

    # Handle list options
    if list_speakers:
        click.echo('Available speakers for custom mode:')
        click.echo('  Vivian       - Bright, slightly edgy young female (Chinese)')
        click.echo('  Serena       - Warm, gentle young female (Chinese)')
        click.echo('  Uncle_Fu     - Seasoned male, low, mellow timbre (Chinese)')
        click.echo('  Dylan        - Youthful Beijing male, clear (Chinese - Beijing)')
        click.echo('  Eric         - Lively Chengdu male, husky (Chinese - Sichuan)')
        click.echo('  Ryan         - Dynamic male, strong rhythmic drive (English)')
        click.echo('  Aiden        - Sunny American male, clear midrange (English)')
        click.echo('  Ono_Anna     - Playful Japanese female, light (Japanese)')
        click.echo('  Sohee        - Warm Korean female, rich emotion (Korean)')
        return

    if list_languages:
        click.echo('Supported languages:')
        click.echo('  Auto, Chinese, English, Japanese, Korean,')
        click.echo('  German, French, Russian, Portuguese, Spanish, Italian')
        return

    # Require text argument if not using list options
    if not text:
        click.echo('Error: TEXT argument is required', err=True)
        click.echo("Try 'qtts --help' for help.", err=True)
        sys.exit(1)

    # Validate mode-specific requirements
    if mode == 'clone':
        if not ref_audio or not ref_text:
            click.echo('Error: clone mode requires --ref-audio and --ref-text', err=True)
            sys.exit(1)

    if mode == 'design':
        if not instruct:
            click.echo('Error: design mode requires --instruct (voice description)', err=True)
            sys.exit(1)

    if speed <= 0:
        click.echo('Error: --speed must be > 0', err=True)
        sys.exit(1)

    # Determine device
    if device is None:
        device = get_device()

    click.echo('=== Qwen3-TTS Speech Generation ===')
    click.echo(f'Mode: {mode}')
    click.echo(f'Text: {text}')
    click.echo(f'Language: {language}')
    click.echo(f'Speed: {speed}%')
    click.echo('')

    # Generate speech based on mode
    if mode == 'clone':
        wav_data, sample_rate = generate_clone_mode(
            text, ref_audio, ref_text, language, device, model
        )
    elif mode == 'custom':
        wav_data, sample_rate = generate_custom_mode(
            text, speaker, language, instruct, device, model
        )
    elif mode == 'design':
        wav_data, sample_rate = generate_design_mode(
            text, instruct, language, device, model
        )
    else:
        click.echo(f'Unknown mode: {mode}', err=True)
        sys.exit(1)

    # Save as WAV first
    output_path = Path(output)
    base_wav = output_path.with_suffix('.wav')

    click.echo('\nSaving audio...')
    sf.write(str(base_wav), wav_data, sample_rate)

    # Apply speed into a temp wav
    processed_wav = output_path.with_suffix('.processed.wav')
    try:
        apply_speed_percent(str(base_wav), str(processed_wav), speed)
    except Exception as e:
        click.echo(f'Error applying speed: {e}', err=True)
        sys.exit(1)
    finally:
        if base_wav.exists():
            try:
                os.remove(base_wav)
            except Exception:
                pass

    # Convert to MP3 or keep as WAV
    if output_path.suffix.lower() == '.mp3':
        click.echo('Converting to MP3...')
        save_as_mp3(str(processed_wav), str(output_path))
        click.echo(f'\n✓ Success! Audio saved to: {output_path}')
    else:
        if processed_wav != output_path:
            os.replace(processed_wav, output_path)
        click.echo(f'\n✓ Success! Audio saved to: {output_path}')


if __name__ == '__main__':
    main()