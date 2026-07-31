#!/usr/bin/env python3

import argparse
import time
from pathlib import Path


DEFAULT_PROMPT = (
    "A highly detailed cinematic photograph of a majestic orange Maine Coon cat "
    "sitting inside an old Japanese ramen shop at night, wearing a tiny dark green "
    "raincoat, one paw resting beside a steaming ceramic bowl, warm paper lanterns "
    "illuminating its individual wet fur strands, neon signs reflected in the "
    "rain-covered window behind it, pedestrians with transparent umbrellas softly "
    "blurred outside, intricate wooden interior, rising volumetric steam, realistic "
    "whiskers and amber eyes, shallow depth of field, natural film grain, dramatic "
    "warm and cool color contrast, physically accurate reflections, photorealistic, "
    "editorial photography, 85mm lens, f/1.8, exceptional detail"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Z-Image with Diffusers")
    parser.add_argument("--model", default="/workspace/models/Z-Image")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--cfg-normalization", action="store_true")
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/diffusers-cinematic-cat-1024.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from diffusers import ZImagePipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Loading pipeline (excluded from generation timing)...")
    pipe = ZImagePipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.cuda.synchronize()
    generation_started = time.perf_counter()
    with torch.inference_mode():
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            cfg_normalization=args.cfg_normalization,
            max_sequence_length=args.seqlen,
            generator=generator,
        ).images[0]
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_started

    save_started = time.perf_counter()
    image.save(args.output)
    save_seconds = time.perf_counter() - save_started

    print(f"Generation time: {generation_seconds:.3f}s")
    print(f"PNG save time:   {save_seconds:.3f}s")
    print(f"Total time:      {generation_seconds + save_seconds:.3f}s")
    print(f"Output:          {args.output}")


if __name__ == "__main__":
    main()
