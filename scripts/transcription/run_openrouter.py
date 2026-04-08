#!/usr/bin/env python3
"""Transcribe manuscript pages using VLMs via OpenRouter.

Supports multiple vision-language models through a single API:
  - GPT-5.4 (OpenAI)
  - Claude Opus 4.6 (Anthropic)
  - Gemini 3.1 Pro (Google)
  - Mistral Large 3 (Mistral)

Requires: OPENROUTER_API_KEY environment variable
Install:  pip install openai

Usage:
  export OPENROUTER_API_KEY="sk-or-..."
  python scripts/transcription/run_openrouter.py --model gpt_5_4
  python scripts/transcription/run_openrouter.py --model claude_opus_4_6 --subset toledo
  python scripts/transcription/run_openrouter.py --list-models

Image Resolution Behavior Per Provider (via OpenRouter)
-------------------------------------------------------
Each provider processes images differently behind OpenRouter's unified API.
This directly impacts CER — image resolution is a key factor in HTR accuracy.

  Provider   | Max native res           | CODEA effective     | Toledo effective     | detail param
  -----------|--------------------------|---------------------|----------------------|---------------
  OpenAI     | 2048px longest → 512 tiles | ~2048px (tiled)    | ~1872px (tiled)      | "high" required
  Anthropic  | 1568px longest side      | ~1568px (6x loss)   | ~1568px (minimal)    | Ignored (auto)
  Google     | 3072px native            | ~3072px (2x loss)   | Full res             | Auto high-res
  Mistral    | Variable (Pixtral enc)   | High res            | Full res             | N/A

  - OpenAI: `detail: "high"` scales longest side to 2048px, then tiles into 512x512
    patches (~170 tokens/tile). Without it, images resize to 512x512 (unusable for HTR).
  - Anthropic: Auto-resizes to fit 1568px on longest side. CODEA images (3000-4218px)
    lose significant resolution. Toledo images (~1900px) are nearly unchanged.
  - Google: Supports up to 3072x3072 natively. Best resolution preservation overall.
    Flat ~258 tokens/image regardless of resolution.
  - Mistral: Pixtral vision encoder handles variable resolution without forced
    downsampling. Token cost scales with image size (~1 token per 16x16 patch).

  All models receive `detail: "high"` — providers that don't use it simply ignore it.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (two levels up from this script)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASELINE_PROMPT, encode_image_base64, run_model

MODEL_REGISTRY = {
    "gpt_5_4": {
        "model_id": "openai/gpt-5.4",
        "rate_limit_delay": 1.0,
    },
    "claude_opus_4_6": {
        "model_id": "anthropic/claude-opus-4.6",
        "rate_limit_delay": 1.0,
        "max_b64_bytes": 5_000_000,  # Anthropic 5 MB base64 limit
    },
    "gemini_3_1_pro": {
        "model_id": "google/gemini-3.1-pro-preview",
        "rate_limit_delay": 4.0,
    },
    "mistral_large_3": {
        "model_id": "mistralai/mistral-large-2512",
        "rate_limit_delay": 1.0,
    },
}


def make_transcriber(max_b64_bytes: int = 0):
    """Create a transcribe function with optional image size limit."""
    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    def transcribe(entry: dict, model_id: str) -> str:
        b64 = encode_image_base64(entry, max_b64_bytes=max_b64_bytes)

        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": BASELINE_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=4096,
        )
        return response.choices[0].message.content

    return transcribe


def list_models():
    print("Available models:\n")
    for key, info in MODEL_REGISTRY.items():
        print(f"  {key:<20s}  {info['model_id']}")
    print(f"\nUsage: python {sys.argv[0]} --model <model_key>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe manuscript pages using VLMs via OpenRouter"
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to use for transcription",
    )
    parser.add_argument(
        "--subset",
        choices=["codea", "toledo", "all"],
        default="all",
        help="Dataset subset to process (default: all)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    args = parser.parse_args()

    if args.list_models:
        list_models()
        sys.exit(0)

    if not args.model:
        parser.error("--model is required (use --list-models to see options)")

    config = MODEL_REGISTRY[args.model]
    transcribe = make_transcriber(max_b64_bytes=config.get("max_b64_bytes", 0))
    run_model(
        args.model,
        config["model_id"],
        transcribe,
        rate_limit_delay=config["rate_limit_delay"],
        subset=args.subset,
    )
