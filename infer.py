from pipelines.pipeline import FocusDiTPipeline
from models.mochivae import MochiVAEWrapper, ae_stride_config
from models.focusdit.modeling_focusdit import FocusDiT
from utils.vis_mask import vis_mask

import torchvision
import torch
import os
from transformers import AutoTokenizer, T5EncoderModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from PIL import Image


def load_pipeline(transformer_path, time_shift, weight_dtype=torch.bfloat16, cache_dir='./cache', device='cuda'):
    vae = MochiVAEWrapper('/fangxueji/Models/genmo/mochi-1-preview/vae').to(device)
    vae.vae.to(device, dtype=torch.float32)
    ae_stride_t, ae_stride_h, ae_stride_w = ae_stride_config['Mochi_D12_6x8x8']
    vae.vae_scale_factor = (ae_stride_t, ae_stride_h, ae_stride_w)
    vae.latents_mean = vae.latents_mean.to(device, dtype=torch.float32)
    vae.latents_std = vae.latents_std.to(device, dtype=torch.float32)

    text_encoder = T5EncoderModel.from_pretrained('/fangxueji/Models/google/flan-t5-xxl', cache_dir=cache_dir, torch_dtype=weight_dtype)
    tokenizer = AutoTokenizer.from_pretrained('/fangxueji/Models/google/flan-t5-xxl', cache_dir=cache_dir)
    scheduler = FlowMatchEulerDiscreteScheduler(shift=time_shift)

    transformer = FocusDiT.from_pretrained(transformer_path, cache_dir=cache_dir, mask_threshold=1.0, device=device, torch_dtype=weight_dtype)
    pipeline = FocusDiTPipeline(tokenizer=tokenizer,
                            text_encoder=text_encoder,
                            vae=vae,
                            transformer=transformer,
                            scheduler=scheduler).to(device)
    pipeline.transformer.eval()
    return pipeline


pipe = load_pipeline('outputs/q_wo_min/checkpoint-77200/model_ema', time_shift=3.0)

negative_prompt = """nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry,
                    """

generator = torch.Generator(device='cpu')
generator.manual_seed(20241120)

prompts = [
    "There is a man who appears to be a rugged adventurer, reminiscent of indiana jones. he is wearing a brown cowboy hat that sits atop his head, casting a shadow over his eyes. his facial features are accentuated by a well-groomed beard and mustache, adding to his adventurous persona. he is dressed in a brown leather vest over a white shirt, which is complemented by a necklace around his neck. the background is blurred, but it gives the impression of a rocky, possibly desert-like environment, which aligns with the adventurous theme. the man's gaze is directed straight at the viewer, suggesting a sense of engagement or challenge.",
]

save_dir = './samples'
os.makedirs(save_dir, exist_ok=True)

for i, prompt in enumerate(prompts):
    pipe_out = pipe(prompt,
                    negative_prompt=negative_prompt,
                    height=512,
                    width=512,
                    num_frames=1,
                    num_inference_steps=30,
                    guidance_scale=5.5,
                    max_sequence_length=512,
                    annealing_factor=None,
                    generator=generator
                )
    image = pipe_out.images[0]
    mask_all = pipe_out.mask_all
    q_mask, k_mask = vis_mask(mask_all, latent_height=32)
    if q_mask is not None:
        torchvision.utils.save_image(q_mask, f'{save_dir}/q_{i}.png')

    save_path = f'{save_dir}/{i}.png'
    image = Image.fromarray(image[0].cpu().numpy().astype('uint8'))
    image.save(save_path)
