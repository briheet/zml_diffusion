# Z-Image Diffusers benchmark

Run from the repository root:

```bash
cd others/diffusers/zimage
uv run main.py \
  --model=/workspace/models/Z-Image \
  --prompt="A highly detailed cinematic photograph of a majestic orange Maine Coon cat sitting inside an old Japanese ramen shop at night, wearing a tiny dark green raincoat, one paw resting beside a steaming ceramic bowl, warm paper lanterns illuminating its individual wet fur strands, neon signs reflected in the rain-covered window behind it, pedestrians with transparent umbrellas softly blurred outside, intricate wooden interior, rising volumetric steam, realistic whiskers and amber eyes, shallow depth of field, natural film grain, dramatic warm and cool color contrast, physically accurate reflections, photorealistic, editorial photography, 85mm lens, f/1.8, exceptional detail" \
  --height=1024 \
  --width=1024 \
  --steps=48 \
  --guidance-scale=4.5 \
  --seqlen=512 \
  --output=/workspace/diffusers-cinematic-cat-1024.png
```

The output prints CUDA-synchronized generation time, PNG save time, and total time.
