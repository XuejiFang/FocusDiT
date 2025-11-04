from mapledit.sample.pipeline_opensora import OpenSoraPipeline
from mapledit.models.causalvideovae import CausalVAEModelWrapper, ae_stride_config
from mapledit.models.diffusion.mapledit.modeling_opensora import MapleT2V
from mapledit.utils.vis_mask import vis_mask

import torchvision, torch, os
from torch.utils.data import Dataset
from transformers import MT5EncoderModel, AutoTokenizer, T5EncoderModel
from diffusers.schedulers import DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_gif
from PIL import Image

import accelerate
from accelerate import Accelerator


def load_pipeline(transformer_path, time_shift, weight_dtype=torch.bfloat16, cache_dir='./cache', device='cuda'):
    vae = CausalVAEModelWrapper('/fangxueji/Models/genmo/mochi-1-preview/vae', ae='Mochi_D12_6x8x8').to(device)
    # vae.vae.enable_tiling()
    # vae.vae.tile_overlap_factor = 0.125
    # vae.vae.tile_sample_min_size = 512
    # vae.vae.tile_latent_min_size = 64
    # vae.vae.tile_sample_min_size_t = 29
    # vae.vae.tile_latent_min_size_t = 8
    # vae.vae_scale_factor = ae_stride_config['CausalVAEModel_D4_4x8x8']
    vae.vae.to(device, dtype=torch.float32)
    ae_stride_t, ae_stride_h, ae_stride_w = ae_stride_config['Mochi_D12_6x8x8']
    vae.vae_scale_factor = (ae_stride_t, ae_stride_h, ae_stride_w)
    vae.latents_mean = vae.latents_mean.to(device, dtype=torch.float32)
    vae.latents_std = vae.latents_std.to(device, dtype=torch.float32)

    text_encoder = T5EncoderModel.from_pretrained('/fangxueji/Models/google/flan-t5-xxl', cache_dir=cache_dir, torch_dtype=weight_dtype)
    tokenizer = AutoTokenizer.from_pretrained('/fangxueji/Models/google/flan-t5-xxl', cache_dir=cache_dir)
    scheduler = FlowMatchEulerDiscreteScheduler(shift=time_shift)
    print(f"\n\ntime_shift={time_shift}\n\n")

    transformer = MapleT2V.from_pretrained(transformer_path, cache_dir=cache_dir, mask_threshold=1.0, device=device, torch_dtype=weight_dtype)
    pipeline = OpenSoraPipeline(tokenizer=tokenizer, 
                            text_encoder=text_encoder,
                            vae=vae,
                            transformer=transformer,
                            scheduler=scheduler).to(device)
    pipeline.transformer.eval()
    return pipeline


pipe = load_pipeline('outputs/q_wo_min/checkpoint-77200/model_ema', time_shift=3.0)
# pipe.to('cuda')

negative_prompt = """nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, 
                    """

# negative_prompt = " "

generator = torch.Generator(device='cpu')
generator.manual_seed(20241120)
# generator = None

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
    # for image
    image = pipe_out.images[0]
    mask_all = pipe_out.mask_all
    q_mask, k_mask = vis_mask(mask_all, latent_height=32)
    if q_mask is not None:
            torchvision.utils.save_image(q_mask, f'{save_dir}/q_{i}.png')

    save_path = f'{save_dir}/{i}.png'
    image = Image.fromarray(image[0].cpu().numpy().astype('uint8'))
    image.save(save_path)
