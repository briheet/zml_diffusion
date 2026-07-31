# ZML Diffusion

## Tested Environment

- RunPod image: `runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster`
- Pod: `a9dk3g7cny`
- GPU: 1× NVIDIA RTX 5090
- CPU: 21 vCPUs (AMD EPYC 9354)
- Memory: 125 GB
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
  --height=1024 \
  --width=1024 \
  --steps=48 \
  --guidance-scale=4.5 \
  --seqlen=512 \
  --output=/workspace/cinematic-cat-1024.png
```

## Benchmark

RTX 5090, 1024×1024, 48 steps; model loading excluded:

| Implementation | Generation + PNG |
|---|---:|
| ZML | 35.091s |
| Diffusers | 40.645s |

ZML was **1.16× faster** in this run. See the
[Diffusers benchmark](others/diffusers/zimage/README.md) for its command.
