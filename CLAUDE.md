# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **FocusDiT** (or MapleEdit), a video/image generation model based on diffusion transformers with attention masking capabilities. The project implements a text-to-video/image pipeline using:
- Custom Diffusion Transformer architecture (MapleT2V) with RoPE positional embeddings
- Causal Video VAE or Mochi VAE for latent encoding/decoding
- T5 text encoder for prompt conditioning
- Flow matching scheduler for diffusion process

The model features adaptive attention masking mechanisms (q_mask, k_mask) that can focus generation on specific regions during inference.

## Code Architecture

### Core Components

1. **Diffusion Model** (`mapledit/models/diffusion/mapledit/`)
   - `modeling_opensora.py`: Main transformer model `MapleT2V` implementing the diffusion backbone
     - Supports two model sizes: MapleT2V-ROPE-S/122 (28 layers) and MapleT2V-ROPE-L/122 (32 layers)
     - Features adaptive FFN dimensions that expand in middle layers
     - Implements progressive masking through transformer blocks
   - `modules.py`: Core building blocks including `BasicTransformerBlock`, `PatchEmbed2D`, attention mechanisms
   - `rope.py`: Rotary Position Embeddings (RoPE) for 3D positional encoding

2. **VAE Models** (`mapledit/models/`)
   - `causalvideovae/`: Causal Video VAE implementation with 3D convolutions
     - Supports multiple stride configurations: 2x8x8, 4x8x8
     - Includes evaluation metrics (FVD, LPIPS, SSIM, PSNR, CLIP score)
   - `mochivae/`: Mochi VAE (6x8x8 stride) - alternative VAE backend
   - Both VAEs use `ae_stride_config` and `ae_channel_config` for latent space configuration

3. **Text Encoders** (`mapledit/models/text_encoder/`)
   - `t5.py`: T5 (FLAN-T5-XXL) encoder for text conditioning
   - `clip.py`: CLIP encoder support

4. **Pipeline** (`mapledit/sample/pipeline_opensora.py`)
   - `OpenSoraPipeline`: Main inference pipeline inheriting from diffusers `DiffusionPipeline`
   - Handles text encoding, latent preparation, denoising loop, and VAE decoding
   - Returns `ImagePipelineOutput` with generated images and attention masks

5. **Utilities**
   - `mapledit/utils/vis_mask.py`: Visualizes attention masks (q_mask, k_mask) from generation
   - `mapledit/utils/utils.py`: General utility functions
   - `mapledit/utils/ema.py`: Exponential Moving Average for model weights

### Key Configuration Parameters

**Model Architecture:**
- `use_q_mask`, `use_k_mask`: Enable query/key attention masking
- `use_rope`: Enable rotary position embeddings (always True for this model)
- `mask_threshold`: Threshold for binarizing attention masks
- `use_softmask`: Use soft (continuous) vs hard (binary) masks
- `use_timestep`: Condition masking on diffusion timestep

**Generation Parameters:**
- `num_frames`: Number of video frames (set to 1 for images)
- `height`, `width`: Output resolution (must be divisible by 8)
- `time_shift`: Flow matching scheduler shift parameter (typically 3.0)
- `guidance_scale`: CFG guidance strength (typically 5.5)
- `num_inference_steps`: Denoising steps (typically 30)
- `annealing_factor`: Optional mask annealing during generation

## Running Inference

The main inference script is `infer.py`. To run:

```bash
python infer.py
```

**Key paths to configure in `infer.py`:**
- Line 18: VAE model path (currently `/fangxueji/Models/genmo/mochi-1-preview/vae`)
- Line 32-33: T5 text encoder path (currently `/fangxueji/Models/google/flan-t5-xxl`)
- Line 47: Transformer checkpoint path (currently `outputs/q_wo_min/checkpoint-77200/model_ema`)

**Pipeline Configuration:**
- Modify `num_frames` (line 71) for video length (1 for images)
- Adjust `height` and `width` (lines 69-70) for resolution
- Change `guidance_scale` (line 73) and `num_inference_steps` (line 72) for quality vs speed
- Set `generator` seed (line 56) for reproducibility

**Output:**
- Generated images saved to `./samples/` directory
- Attention mask visualizations saved as `q_{i}.png` if masking is enabled

## Model Checkpoints

Checkpoints are stored in `outputs/` directory with structure:
```
outputs/{experiment_name}/checkpoint-{step}/model_ema/
├── config.json
└── diffusion_pytorch_model.safetensors.index.json
```

EMA (Exponential Moving Average) weights are recommended for inference (`model_ema/` subdirectory).

## VAE Configuration

The codebase supports multiple VAE backends via `ae_stride_config`:
- `CausalVAEModel_D4_4x8x8`: 4 channels, 4x8x8 downsampling
- `Mochi_D12_6x8x8`: 12 channels, 6x8x8 downsampling (default in infer.py)

Latent dimensions are computed as: `(T÷stride_t, H÷stride_h, W÷stride_w)`

## Attention Masking System

The model's unique feature is adaptive attention masking during generation:

1. **Mask Propagation**: Masks are computed in each transformer block and passed to subsequent blocks via `mask_q_pre`
2. **Mask Types**:
   - `q_mask`: Query masking (which tokens attend)
   - `k_mask`: Key masking (which tokens are attended to)
3. **Visualization**: Use `vis_mask()` from `mapledit/utils/vis_mask.py` to visualize mask evolution across timesteps
4. **Usage**: Masks returned in `pipe_out.mask_all` can be saved/analyzed

## Dependencies

Core dependencies (no requirements.txt in repo):
- PyTorch with CUDA support
- diffusers (Hugging Face)
- transformers (Hugging Face)
- accelerate
- torchvision
- einops
- PIL

VAE evaluation dependencies (in `mapledit/models/causalvideovae/eval/`):
- LPIPS, FVD, SSIM, PSNR metrics
- Evaluation scripts in `eval/script/*.sh`

## Important Notes

- This is a research codebase for FocusDiT - the model uses hard-coded paths to external model checkpoints (VAE, T5)
- The model works with both image (num_frames=1) and video (num_frames>1) generation
- Latent frame computation handles odd frame counts: `((frame-1)//stride_t + 1) if frame%2==1 else frame//stride_t`
- The pipeline uses flow matching (Euler discrete scheduler) rather than traditional DDPM
- Negative prompts are important for quality - see example in `infer.py:50-51`
