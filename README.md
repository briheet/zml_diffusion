# ZML Diffusion

## Tested Environment

- RunPod image: `runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster`
- Pod: `a9dk3g7cny`
- GPU: 1× NVIDIA RTX 5090
- Driver: 580.159.03 (CUDA 13.0)
- CPU: 76 vCPUs (AMD EPYC 9J14 96-Core Processor)
- Memory: 150 GB
- Container disk: 50 GB

## Install Nix on Ubuntu

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install curl -y

sh <(curl -L https://nixos.org/nix/install) --daemon
. /etc/profile.d/nix.sh

sudo mkdir -p /etc/nix
echo "experimental-features = nix-command flakes" |
  sudo tee -a /etc/nix/nix.conf
```

Restart the shell, then verify the installation:

```bash
nix --version
nix-shell -p hello --run hello
nix flake --help
```

## Run

Download the model:

```bash
HF_XET_HIGH_PERFORMANCE=1 \
  uvx --from huggingface-hub hf download Tongyi-MAI/Z-Image \
  --local-dir /workspace/models/Z-Image
```

Generate an image on an NVIDIA Linux machine:

```bash
nix run .#cudazimage -- \
  --model=/workspace/models/Z-Image \
  --prompt="A highly detailed cinematic photograph of a majestic orange Maine Coon cat sitting inside an old Japanese ramen shop at night, wearing a tiny dark green raincoat, one paw resting beside a steaming ceramic bowl, warm paper lanterns illuminating its individual wet fur strands, neon signs reflected in the rain-covered window behind it, pedestrians with transparent umbrellas softly blurred outside, intricate wooden interior, rising volumetric steam, realistic whiskers and amber eyes, shallow depth of field, natural film grain, dramatic warm and cool color contrast, physically accurate reflections, photorealistic, editorial photography, 85mm lens, f/1.8, exceptional detail" \
  --negative-prompt="blurry, low quality, distorted anatomy, text, watermark" \
  --height=1024 \
  --width=1024 \
  --steps=48 \
  --guidance-scale=4.5 \
  --seqlen=512 \
  --output=/workspace/cinematic-cat-1024.png
```

## Benchmark

RTX 5090, 1024×1024, 48 steps; model loading excluded. Results are the mean of
four runs:

| Implementation | Mean generation + PNG |
|---|---:|
| ZML | 35.012s |
| Diffusers | 41.154s |

ZML was **1.18× faster** on average. See the 
[Diffusers benchmark](others/diffusers/zimage/README.md) for diffusers configuration.

## Development

```bash
nix develop
hx build.zig src/main.zig
```

The shell provides a Bazel-backed ZLS with navigation and completion for ZML.
Opening `build.zig` primes ZLS's Bazel module map. The dependency remains in
Bazel's external repository cache; it is not vendored.
