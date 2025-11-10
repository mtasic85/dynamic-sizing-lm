#!/usr/bin/env python3
"""
Dynamic Sizing Language Model (DSLM) CLI Tool

A tool for upscaling and downscaling language models using the HyperCloning method.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from common import count_parameters, format_parameter_count  # type: ignore
from up import upscale_model  # type: ignore
from down import downscale_model  # type: ignore


def cmd_up(args):
    """Handle the 'up' subcommand for upscaling models."""
    try:
        print(f"Upscaling model: {args.input}")
        print(f"Embedding dimension multiplier: {args.embed_dim_multiplier}")
        print(f"Up projection multiplier: {args.up_proj_multiplier}")

        # Validate multipliers are integers
        if (
            not isinstance(args.embed_dim_multiplier, int)
            or args.embed_dim_multiplier < 1
        ):
            raise ValueError("embed_dim_multiplier must be a positive integer")
        if not isinstance(args.up_proj_multiplier, int) or args.up_proj_multiplier < 1:
            raise ValueError("up_proj_multiplier must be a positive integer")

        # Load and describe input model
        print("\n" + "=" * 60)
        print("INPUT MODEL DESCRIPTION")
        print("=" * 60)
        input_model = AutoModelForCausalLM.from_pretrained(
            args.input, trust_remote_code=True, dtype=torch.float32
        )
        print(input_model)
        input_param_count = count_parameters(input_model)
        print(
            f"\nInput model total parameters: {format_parameter_count(input_param_count)}"
        )

        # Upscale the model
        print("\n" + "=" * 60)
        print("UPSCALING...")
        print("=" * 60)
        upscaled_model, output_path = upscale_model(
            model_path=args.input,
            embed_dim_multiplier=args.embed_dim_multiplier,
            up_proj_multiplier=args.up_proj_multiplier,
            output_path=args.output,
            snr_db=getattr(args, "snr_db", None),
        )

        # Save the model
        print(f"Saving upscaled model to: {output_path}")
        upscaled_model.save_pretrained(output_path)

        # Also save tokenizer if it exists
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.input)
            tokenizer.save_pretrained(output_path)
            print("Tokenizer saved successfully")
        except Exception as e:
            print(f"Warning: Could not save tokenizer: {e}")

        # Load and describe output model
        print("\n" + "=" * 60)
        print("OUTPUT MODEL DESCRIPTION")
        print("=" * 60)
        output_model = AutoModelForCausalLM.from_pretrained(
            output_path, trust_remote_code=True, dtype=torch.float32
        )
        print(output_model)
        output_param_count = count_parameters(output_model)
        print(
            f"\nOutput model total parameters: {format_parameter_count(output_param_count)}"
        )

        print("\n" + "=" * 60)
        print("UPSCALING COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"Error during upscaling: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_down(args):
    """Handle the 'down' subcommand for downscaling models."""
    try:
        print(f"Downscaling model: {args.input}")

        # Load and describe input model
        print("\n" + "=" * 60)
        print("INPUT MODEL DESCRIPTION")
        print("=" * 60)
        input_model = AutoModelForCausalLM.from_pretrained(
            args.input, trust_remote_code=True, dtype=torch.float32
        )
        print(input_model)
        input_param_count = count_parameters(input_model)
        print(
            f"\nInput model total parameters: {format_parameter_count(input_param_count)}"
        )

        # This will raise NotImplementedError for now
        print("\n" + "=" * 60)
        print("DOWNSCALING...")
        print("=" * 60)
        downscaled_model, output_path = downscale_model(
            model_path=args.input, output_path=args.output
        )

        # When implemented, this would save and describe the output model
        print(f"Output path would be: {output_path}")

    except Exception as e:
        if isinstance(e, NotImplementedError):
            print(f"Downscaling not yet implemented: {e}", file=sys.stderr)
        else:
            print(f"Error during downscaling: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_desc(args):
    """Handle the 'desc' subcommand for describing models."""
    try:
        print(f"Loading model: {args.input}")
        model = AutoModelForCausalLM.from_pretrained(
            args.input, trust_remote_code=True, dtype=torch.float32
        )

        # Print the model architecture
        print(model)

        # Print total parameter count
        param_count = count_parameters(model)
        print(f"\nTotal parameters: {format_parameter_count(param_count)}")

    except Exception as e:
        print(f"Error describing model: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_gen(args):
    """Handle the 'gen' subcommand for text generation."""
    try:
        print(f"Loading model: {args.input}")
        model = AutoModelForCausalLM.from_pretrained(
            args.input, trust_remote_code=True, dtype=torch.float32
        )
        tokenizer = AutoTokenizer.from_pretrained(args.input)

        # Move model to appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)  # type: ignore
        print(f"Using device: {device}")

        # Prepare input
        prompt = args.prompt or "The capital of France is"
        print(f"Max tokens: {args.n_predict}")

        # Get generation parameters (defaults are set in argument parser)
        temperature = args.temperature
        top_k = args.top_k
        top_p = args.top_p

        print(f"Temperature: {temperature}")
        print(f"Top-k: {top_k}")
        print(f"Top-p: {top_p}")

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Determine if we should use sampling
        do_sample = temperature > 0.0 or top_p < 1.0 or top_k is not None

        # Print initial prompt
        print(prompt, end="", flush=True)

        # Create streamer for incremental output
        streamer = TextStreamer(tokenizer, skip_prompt=True)

        # Generate with streaming
        with torch.no_grad():
            model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=args.n_predict,
                num_return_sequences=1,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_k=top_k if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=tokenizer.eos_token_id,
                streamer=streamer,
            )

    except Exception as e:
        print(f"Error during generation: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Dynamic Sizing Language Model (DSLM) Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upscale a Qwen3-0.6B model
  dslm up --input Qwen/Qwen3-0.6B --embed-dim-multiplier 2 --up-proj-multiplier 2

  # Describe a model
  dslm desc --input HuggingFaceTB/SmolLM2-360M

  # Generate text
  dslm gen --input Qwen/Qwen3-0.6B --prompt "Hello world" --n-predict 50

   # Downscale (not yet implemented)
   dslm down --input large-model --output small-model
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input model path or HuggingFace identifier",
    )

    # Up command
    up_parser = subparsers.add_parser(
        "up", parents=[common_parser], help="Upscale a model using HyperCloning"
    )
    up_parser.add_argument(
        "-o",
        "--output",
        help="Output path for upscaled model (auto-generated if not provided)",
    )
    up_parser.add_argument(
        "-edm",
        "--embed-dim-multiplier",
        type=int,
        required=True,
        help="Integer multiplier for embedding dimensions",
    )
    up_parser.add_argument(
        "-upm",
        "--up-proj-multiplier",
        type=int,
        required=True,
        help="Integer multiplier for FFN dimensions",
    )
    up_parser.add_argument(
        "--snr-db", type=float, help="Signal-to-noise ratio for adding noise (optional)"
    )
    up_parser.set_defaults(func=cmd_up)

    # Down command
    down_parser = subparsers.add_parser(
        "down", parents=[common_parser], help="Downscale a model (not yet implemented)"
    )
    down_parser.add_argument("-o", "--output", help="Output path for downscaled model")
    down_parser.set_defaults(func=cmd_down)

    # Desc command
    desc_parser = subparsers.add_parser(
        "desc",
        parents=[common_parser],
        help="Describe model architecture and parameters",
    )
    desc_parser.set_defaults(func=cmd_desc)

    # Gen command
    gen_parser = subparsers.add_parser(
        "gen", parents=[common_parser], help="Generate text using the model"
    )

    gen_parser.add_argument(
        "-p",
        "--prompt",
        default="The capital of France is",
        help="Prompt for text generation",
    )
    gen_parser.add_argument(
        "-n",
        "--n-predict",
        type=int,
        default=100,
        help="Number of tokens to generate (default: 100)",
    )
    gen_parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    gen_parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k sampling (default: 40)",
    )
    gen_parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling (default: 0.9)",
    )
    gen_parser.set_defaults(func=cmd_gen)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Run the appropriate command
    args.func(args)


if __name__ == "__main__":
    main()
