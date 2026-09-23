"""Generate an image with the released FocusDiT checkpoint."""

import argparse
from pathlib import Path

import torch
import torchvision
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel

from models.focusdit.modeling_focusdit import FocusDiT
from models.mochivae import MochiVAEWrapper, ae_stride_config
from pipelines.pipeline import FocusDiTPipeline
from utils.vis_mask import vis_mask


DEFAULT_PROMPT = (
    "There is a man who appears to be a rugged adventurer, reminiscent of indiana jones. "
    "he is wearing a brown cowboy hat that sits atop his head, casting a shadow over his eyes. "
    "his facial features are accentuated by a well-groomed beard and mustache, adding to his "
    "adventurous persona. he is dressed in a brown leather vest over a white shirt, which is "
    "complemented by a necklace around his neck. the background is blurred, but it gives the "
    "impression of a rocky, possibly desert-like environment, which aligns with the adventurous "
    "theme. the man's gaze is directed straight at the viewer, suggesting a sense of engagement "
    "or challenge."
)
DEFAULT_NEGATIVE_PROMPT = (
    "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, "
    "extra digit, fewer digits, cropped, worst quality, low quality, normal quality, "
    "jpeg artifacts, signature, watermark, username, blurry"
)


def load_pipeline(transformer_path, vae_path, text_encoder_path, time_shift=3.0,
                  weight_dtype=torch.bfloat16, device="cuda", text_encoder_device="cuda",
                  vae_device="cuda"):
    vae = MochiVAEWrapper(str(vae_path)).to(vae_device)
    vae.vae.to(vae_device, dtype=torch.float32)
    vae.vae_scale_factor = tuple(ae_stride_config["Mochi_D12_6x8x8"])
    vae.latents_mean = vae.latents_mean.to(vae_device, dtype=torch.float32)
    vae.latents_std = vae.latents_std.to(vae_device, dtype=torch.float32)

    text_encoder = T5EncoderModel.from_pretrained(
        str(text_encoder_path), torch_dtype=weight_dtype
    ).to(text_encoder_device)
    tokenizer = AutoTokenizer.from_pretrained(str(text_encoder_path))
    scheduler = FlowMatchEulerDiscreteScheduler(shift=time_shift)
    transformer = FocusDiT.from_pretrained(
        str(transformer_path), mask_threshold=1.0, device=device, torch_dtype=weight_dtype
    ).to(device)
    pipeline = FocusDiTPipeline(
        tokenizer=tokenizer, text_encoder=text_encoder, vae=vae,
        transformer=transformer, scheduler=scheduler,
    )
    pipeline.transformer.eval()
    return pipeline


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-path", type=Path, required=True,
                        help="Local directory containing the FocusDiT Diffusers config and weights")
    parser.add_argument("--vae-path", type=Path, required=True,
                        help="Local Mochi VAE directory (the vae/ subfolder)")
    parser.add_argument("--text-encoder-path", type=Path, required=True,
                        help="Local google/flan-t5-xxl model directory")
    parser.add_argument("--text-encoder-device", choices=("cuda", "cpu"), default="cuda",
                        help="Use CPU when GPU memory cannot hold the T5 encoder")
    parser.add_argument("--vae-device", choices=("cuda", "cpu"), default="cuda",
                        help="Use CPU to reduce GPU memory use during VAE decoding")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=Path("samples"))
    parser.add_argument("--seed", type=int, default=20241120)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.5)
    parser.add_argument("--save-mask", action="store_true", help="Save a visualization of the query masks")
    return parser.parse_args()


def main():
    args = parse_args()
    for label, path in (("Transformer", args.transformer_path),
                        ("VAE", args.vae_path), ("text encoder", args.text_encoder_path)):
        if not path.is_dir():
            raise SystemExit(f"{label} directory does not exist: {path}")
    if not torch.cuda.is_available():
        raise SystemExit("This inference example requires a CUDA GPU.")
    if args.height <= 0 or args.width <= 0 or args.height % 16 or args.width % 16:
        raise SystemExit("Height and width must be positive multiples of 16 pixels.")
    if args.steps <= 0:
        raise SystemExit("Steps must be a positive integer.")

    pipe = load_pipeline(
        args.transformer_path, args.vae_path, args.text_encoder_path,
        text_encoder_device=args.text_encoder_device, vae_device=args.vae_device,
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    result = pipe(
        args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=1,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=512,
        generator=generator,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(result.images[0][0].cpu().numpy().astype("uint8"))
    image.save(args.output_dir / "image.png")
    if args.save_mask:
        # One image patch spans 16 pixels (8x VAE downsampling, 2x transformer patch).
        q_mask, _ = vis_mask(result.mask_all, latent_height=args.height // 16)
        if q_mask is not None:
            torchvision.utils.save_image(q_mask, args.output_dir / "query_mask.png")


if __name__ == "__main__":
    main()
