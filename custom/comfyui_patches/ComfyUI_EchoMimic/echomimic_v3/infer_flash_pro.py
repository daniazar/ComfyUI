import os
import sys
import argparse
import gc
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer
import folder_paths
import torchvision.transforms.functional as TF
#from .src.dist import set_multi_gpus_devices, shard_model
from .src.wan_vae import AutoencoderKLWan
from .src.wan_image_encoder import  CLIPModel
from .src.wan_text_encoder import  WanT5EncoderModel
from .src.wan_transformer3d_audio_2512 import WanTransformerAudioMask3DModel
from .src.pipeline_wan_fun_inpaint_audio_2512 import WanFunInpaintAudioPipeline

from .src.utils import (filter_kwargs, get_image_to_video_latent, get_image_to_video_latent2,
                                   save_videos_grid)

from .src.fm_solvers import FlowDPMSolverMultistepScheduler
from .src.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .src.cache_utils import get_teacache_coefficients
from .infer import encode_prompt,get_image_to_video_latent3
import decord
import json
import random
import math
import comfy.model_management as mm
import librosa
try:
    from moviepy.editor import  VideoFileClip, AudioFileClip
except:
    try:
        from moviepy import VideoFileClip, AudioFileClip
    except:
        from moviepy import *
import pyloudnorm as pyln
from transformers import Wav2Vec2FeatureExtractor
from  .src.wav2vec2 import Wav2Vec2Model
from einops import rearrange

def clear_comfyui_cache():
    cf_models=mm.loaded_models()
    try:
        for pipe in cf_models:
            pipe.unpatch_model(device_to=torch.device("cpu"))
            print(f"Unpatching models.{pipe}")
    except: pass
    mm.soft_empty_cache()
    torch.cuda.empty_cache()
    max_gpu_memory = torch.cuda.max_memory_allocated()
    print(f"After Max GPU memory allocated: {max_gpu_memory / 1000 ** 3:.2f} GB")


def parse_args():
    parser = argparse.ArgumentParser(description="WanFun Inference")
    
    # Model paths and config
    parser.add_argument("--config_path", type=str, default="config/wan2.1/wan_civitai.yaml", help="Config path")
    parser.add_argument("--model_name", type=str, default="Wan2.1-Fun-V1.1-1.3B-InP", help="Model name")
    parser.add_argument("--ckpt_idx", type=int, default=50000, help="Checkpoint index")
    parser.add_argument("--transformer_path", type=str, default="", help="Transformer path")
    parser.add_argument("--vae_path", type=str, default=None, help="VAE path")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA path")
    parser.add_argument("--save_path", type=str, default="outputs", help="Save path")
    
    # Audio model path
    parser.add_argument("--wav2vec_model_dir", type=str, default="chinese-wav2vec2-base", help="Wav2Vec model directory")
    
    # Input paths
    parser.add_argument("--image_path", type=str, required=True, help="Input image path")
    parser.add_argument("--audio_path", type=str, required=True, help="Input audio path")
    
    # Inference parameters
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--sampler_name", type=str, default="Flow_Unipc", choices=["Flow", "Flow_Unipc", "Flow_DPM++"], help="Sampler name")
    parser.add_argument("--video_length", type=int, default=81, help="Video length")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Guidance scale")
    parser.add_argument("--audio_guidance_scale", type=float, default=3.0, help="Audio guidance scale")
    parser.add_argument("--audio_scale", type=float, default=1.0, help="Audio scale")
    parser.add_argument("--neg_scale", type=float, default=1.0, help="Negative scale")
    parser.add_argument("--neg_steps", type=int, default=0, help="Negative steps")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=43, help="Random seed")
    parser.add_argument("--lora_weight", type=float, default=0.6, help="LoRA weight")
    
    # TeaCache parameters
    parser.add_argument("--enable_teacache", action="store_true", default=True, help="Enable TeaCache")
    parser.add_argument("--teacache_threshold", type=float, default=0.1, help="TeaCache threshold")
    parser.add_argument("--num_skip_start_steps", type=int, default=5, help="Number of skip start steps")
    parser.add_argument("--teacache_offload", action="store_true", default=False, help="TeaCache offload")
    
    # Dynamic CFG
    parser.add_argument("--use_dynamic_cfg", action="store_true", default=False, help="Use dynamic CFG")
    parser.add_argument("--use_dynamic_acfg", action="store_true", default=False, help="Use dynamic audio CFG")
    
    # Riflex
    parser.add_argument("--enable_riflex", action="store_true", default=False, help="Enable Riflex")
    parser.add_argument("--riflex_k", type=int, default=6, help="Riflex k")
    
    # Mask
    parser.add_argument("--use_un_ip_mask", action="store_true", default=False, help="Use un IP mask")
    
    # GPU and memory
    parser.add_argument("--GPU_memory_mode", type=str, default="sequential_cpu_offload", help="GPU memory mode")
    parser.add_argument("--ulysses_degree", type=int, default=1, help="Ulysses degree")
    parser.add_argument("--ring_degree", type=int, default=1, help="Ring degree")
    parser.add_argument("--fsdp_dit", action="store_true", default=False, help="FSDP DIT")
    parser.add_argument("--weight_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"], help="Weight dtype")
    
    # Other parameters
    parser.add_argument("--sample_size", type=int, nargs=2, default=[768, 768], help="Sample size")
    parser.add_argument("--fps", type=int, default=25, help="FPS")
    parser.add_argument("--add_prompt", type=str, default="", help="Additional prompt")
    parser.add_argument("--negative_prompt", type=str, default="Gesture is bad. Gesture is unclear. Strange and twisted hands. Bad hands. Bad fingers. Unclear and blurry hands. Unclear gestures, broken hands, fused fingers. 手指融合，", help="Negative prompt")
    parser.add_argument("--mouth_prompts", type=str, default=None, help="Mouth prompts")
    
    # Skip ratio
    parser.add_argument("--cfg_skip_ratio", type=float, default=0.0, help="CFG skip ratio")
    parser.add_argument("--shift", type=float, default=5.0, help="Shift value")
    
    return parser.parse_args()

def get_sample_size(pil_img, sample_size):
    w, h = pil_img.size
    ori_a = w * h
    default_a = sample_size[0] * sample_size[1]
    if default_a < ori_a:
        ratio_a = math.sqrt(ori_a / sample_size[0] / sample_size[1])

        w = w / ratio_a // 16 * 16
        h = h / ratio_a // 16 * 16
    else:
        w = w // 16 * 16
        h = h // 16 * 16

    return [int(h), int(w)]


def get_ip_mask(coords):
    y1, y2, x1, x2, h, w = coords
    Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    mask = (Y.unsqueeze(-1) >= y1) & (Y.unsqueeze(-1) < y2) & (X.unsqueeze(-1) >= x1) & (X.unsqueeze(-1) < x2)
    
    mask = mask.reshape(-1)
    return mask.float()

def get_audio_embed(mel_input, wav2vec_feature_extractor, audio_encoder, video_length, sr=16000, fps=25, device='cpu'):

    audio_feature = np.squeeze(wav2vec_feature_extractor(mel_input, sampling_rate=sr).input_values)
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
    audio_feature = audio_feature.unsqueeze(0)

    # audio encoder
    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)
        #embeddings = audio_encoder(audio_feature, output_hidden_states=True)

    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")

    audio_emb = audio_emb.cpu().detach()
    return audio_emb

def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio


def load_v3_flash(sampler_name,vae_path,inp_vae,weigths_current_path,config_path,node_dir,use_mmgp,device,fsdp_dit=True,weight_dtype_str="bfloat16",block_offload=False):
    weight_dtype = torch.bfloat16 if weight_dtype_str == "bfloat16" else torch.float16

    # # Load audio models
    # audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec_model_dir, local_files_only=True).to('cpu')
    # audio_encoder.feature_extractor._freeze_parameters()
    # wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_model_dir, local_files_only=True)

    #device = set_multi_gpus_devices(ulysses_degree, ring_degree)
    config = OmegaConf.load(config_path)
    model_name=os.path.join(node_dir,"Wan2.1-Fun-V1.1-1.3B-InP")
    transformer = WanTransformerAudioMask3DModel.from_pretrained(
         os.path.join(weigths_current_path,"transformer"),
        #os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True if not fsdp_dit else False,
        torch_dtype=weight_dtype,
    )
    transformer_path=os.path.join(weigths_current_path,"echomimicv3-flash-pro/diffusion_pytorch_model.safetensors")
    ckpt_idx=50000
    if transformer_path is not None:
        
        print(f"From checkpoint: {transformer_path}")
        if transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(transformer_path)
        else:
            state_dict = torch.load(os.path.join(transformer_path, f'checkpoint-{ckpt_idx}.pth'))
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Get Vae
    vae_path_=folder_paths.get_full_path("vae", vae_path)
    inp_vae_path=folder_paths.get_full_path("vae", inp_vae) if inp_vae != "none" else None
    vae = AutoencoderKLWan.from_pretrained(vae_path_,
        #os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(weight_dtype)

    if inp_vae_path is not None:
        print(f"From checkpoint: {inp_vae_path}")
        if inp_vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(inp_vae_path)
        else:
            state_dict = torch.load(inp_vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    # Get Text encoder
    # text_encoder = WanT5EncoderModel.from_pretrained(
    #     os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    #     additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    #     low_cpu_mem_usage=True,
    #     torch_dtype=weight_dtype,
    # )
    # text_encoder = text_encoder.eval()

    # Get Clip Image Encoder
    # clip_image_encoder = CLIPModel.from_pretrained(
    #     os.path.join(model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    # ).to(weight_dtype)
    # clip_image_encoder = clip_image_encoder.eval()

    # Get Scheduler
    Choosen_Scheduler = scheduler_dict = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[sampler_name]
    if sampler_name == "Flow_Unipc" or sampler_name == "Flow_DPM++":
        config['scheduler_kwargs']['shift'] = 1
    scheduler = Choosen_Scheduler(
        **filter_kwargs(Choosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Get Pipeline
    pipeline = WanFunInpaintAudioPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        #text_encoder=text_encoder,
        scheduler=scheduler,
        #clip_image_encoder=clip_image_encoder
    )
    

    # if ulysses_degree > 1 or ring_degree > 1:
    #     from functools import partial
    #     transformer.enable_multi_gpus_inference()
    #     if fsdp_dit:
    #         shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
    #         pipeline.transformer = shard_fn(pipeline.transformer)


    if use_mmgp!="None":
        from mmgp import offload, profile_type
        pipeline.to("cpu")
        if use_mmgp=="VerylowRAM_LowVRAM":
            offload.profile(pipeline, profile_type.VerylowRAM_LowVRAM,quantizeTransformer=config.quantize_transformer)
        elif use_mmgp=="LowRAM_LowVRAM":  
            offload.profile(pipeline, profile_type.LowRAM_LowVRAM,quantizeTransformer=config.quantize_transformer)
        elif use_mmgp=="LowRAM_HighVRAM":
            offload.profile(pipeline, profile_type.LowRAM_HighVRAM,quantizeTransformer=config.quantize_transformer)
        elif use_mmgp=="HighRAM_LowVRAM":
            offload.profile(pipeline, profile_type.HighRAM_LowVRAM,quantizeTransformer=config.quantize_transformer)
        elif use_mmgp=="HighRAM_HighVRAM":
            offload.profile(pipeline, profile_type.HighRAM_HighVRAM,quantizeTransformer=config.quantize_transformer)
    elif block_offload:
        pipeline.to("cpu")
    else:
        pipeline.to(device)
    temporal_compression_ratio=pipeline.vae.config.temporal_compression_ratio
    return pipeline,temporal_compression_ratio,tokenizer


def infer_flash(pipeline,audio_embeds,prompt_embeds,negative_prompt_embeds,clip_context,fps,num_inference_steps,seed,video_length_actual,device,block_offload,
                input_video,input_video_mask,sample_height,sample_width,guidance_scale,latent_frames,audio_file_prefix,
                partial_video_length=113, overlap_video_length=8, ref_image_pil=None, temporal_compression_ratio=4, shot_boundaries=None, save_parts=False,
                enable_riflex=False,
                enable_teacache=False,teacache_offload=False,teacache_threshold=0.1,riflex_k=6,audio_scale=1.0,use_un_ip_mask=False,
                num_skip_start_steps=5,audio_guidance_scale=3.0,neg_scale=1.0,neg_steps=0,use_dynamic_cfg=False,
                use_dynamic_acfg=False,cfg_skip_ratio=0.0,shift=5.0,
                ):
    coefficients = get_teacache_coefficients("Wan2.1-Fun-V1.1-1.3B-InP") if enable_teacache else None
    if coefficients is not None:
        print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
        pipeline.transformer.enable_teacache(
            coefficients, num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
        )

    generator = torch.Generator(device=device).manual_seed(seed)
    # image_path=""
    # audio_path=""
    # prompt=""
    # if lora_path is not None:
    #     pipeline = merge_lora(pipeline, lora_path, lora_weight)

    #pipeline.to(device=device)

    # Create output directory
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        # Process single image and audio
        # print(f"Processing: {image_path}")
        # print(f"Audio: {audio_path}")
        # print(f"Prompt: {prompt}")
        
        # Generate output filename
        # image_name = os.path.basename(image_path).split('.')[0]
        # output_video_path = os.path.join(folder_paths.output_directory, f"{image_name}_output.mp4")
        
        # Check if output already exists
        # if os.path.exists(output_video_path):
        #     print(f"⏭️  Skip: {output_video_path} already exists.")
        #     return

        # # Load reference image
        # ref_image = Image.open(image_path).convert("RGB")
        # ref_start = np.array(ref_image)

        # # Load audio
        # audio_clip = AudioFileClip(audio_path)
        # video_length_actual = min(int(audio_clip.duration * fps), video_length)
        # video_length_actual = int((video_length_actual - 1) // pipeline.temporal_compression_ratio * pipeline.temporal_compression_ratio) + 1 if video_length_actual != 1 else 1

        # # Get audio features
        # mel_input, sr = librosa.load(audio_path, sr=16000) #TODO: 
        # mel_input = loudness_norm(mel_input, sr)
        # mel_input = mel_input[:int(video_length_actual / 25 * sr)]
        
        # print(f"Audio length: {int(len(mel_input)/ sr * 25)}, Video length: {video_length_actual}")
        # #audio_feature_wav2vec = get_audio_embed(mel_input, wav2vec_feature_extractor, audio_encoder, video_length_actual, sr=16000, fps=25, device='cpu')
        
        # # Get audio batch 
        # audio_embeds = audio_feature_wav2vec.to(device=device, dtype=weight_dtype)
        
        # indices = (torch.arange(2 * 2 + 1) - 2) * 1 
        # center_indices = torch.arange(
        #     0,  
        #     video_length_actual,
        if enable_riflex:
            pipeline.transformer.enable_riflex(k = riflex_k, L_test = latent_frames)
        
        import math
        import comfy.utils

        source_images = ref_image_pil if isinstance(ref_image_pil, list) else [ref_image_pil]
        num_shots = len(source_images)
        
        if shot_boundaries is None:
            shot_boundaries = np.linspace(0, video_length_actual, num_shots + 1, dtype=int).tolist()

        total_parts = 0
        for i in range(num_shots):
            frames_in_shot = shot_boundaries[i+1] - shot_boundaries[i]
            if frames_in_shot <= partial_video_length:
                total_parts += 1
            else:
                total_parts += 1 + math.ceil((frames_in_shot - partial_video_length) / (partial_video_length - overlap_video_length))
                
        total_steps = total_parts * num_inference_steps
        pbar = comfy.utils.ProgressBar(total_steps)
        
        save_parts_to_disk = save_parts # Used from node toggle
        init_frames = 0
        new_sample = None
        part_idx = 0
        active_shot_idx = -1

        while init_frames < video_length_actual:
            current_shot_idx = 0
            for i in range(num_shots):
                if init_frames >= shot_boundaries[i] and init_frames < shot_boundaries[i+1]:
                    current_shot_idx = i
                    break
            
            is_new_shot = (current_shot_idx != active_shot_idx)
            if is_new_shot:
                active_shot_idx = current_shot_idx
                ref_img = source_images[current_shot_idx]
                current_overlap = 0
            else:
                dc = np.array(source_images[current_shot_idx])
                ref_img = []
                for i in range(-overlap_video_length, 0):
                    sc = (sample[0, :, i].transpose(0, 1).transpose(1, 2) * 255).numpy().astype(np.uint8)
                    try:
                        from .src.utils import color_transfer
                        sc = color_transfer(sc, dc)
                    except Exception as e:
                        pass
                    ref_img.append(Image.fromarray(sc))
                current_overlap = overlap_video_length

            frames_left_in_shot = shot_boundaries[current_shot_idx + 1] - init_frames
            
            target_length = min(partial_video_length, frames_left_in_shot + current_overlap)
            target_length = int((target_length - 1) // temporal_compression_ratio * temporal_compression_ratio) + 1 if target_length != 1 else 1
            
            if target_length <= 1:
                # We reached exactly the boundary due to rounding, step into next shot
                init_frames = shot_boundaries[current_shot_idx + 1]
                continue
                
            _input_video, _input_video_mask, _ = get_image_to_video_latent2(
                ref_img, None, video_length=target_length, sample_size=[sample_height, sample_width]
            )

            global_start = init_frames - current_overlap
            global_end = init_frames + target_length - current_overlap
            
            # Pad audio if it's slightly shorter than needed (e.g. at the very end of the video)
            partial_audio_embeds = audio_embeds[:, global_start : global_end]
            if partial_audio_embeds.shape[1] < target_length:
                pad_len = target_length - partial_audio_embeds.shape[1]
                pad_audio = partial_audio_embeds[:, -1:].repeat(1, pad_len, 1, 1, 1)
                partial_audio_embeds = torch.cat([partial_audio_embeds, pad_audio], dim=1)
            
            print(f"EchoMimic: Starting part {part_idx + 1} of {total_parts} (frames {global_start} to {global_end})...")

            pipeline.set_progress_bar_config(disable=True)
            
            def step_callback(pipe, step, timestep, callback_kwargs):
                pbar.update(1)
                global_step = part_idx * num_inference_steps + step + 1
                percent = int(global_step / total_steps * 100)
                if step % 5 == 0 or step == num_inference_steps - 1:
                    print(f"EchoMimic: Rendering part {part_idx + 1} of {total_parts} - Step {step + 1}/{num_inference_steps} ({percent}%)")
                return callback_kwargs

            if isinstance(prompt_embeds, list) and len(prompt_embeds) > 0 and isinstance(prompt_embeds[0], list):
                current_prompt_embeds = prompt_embeds[min(current_shot_idx, len(prompt_embeds) - 1)]
            else:
                current_prompt_embeds = prompt_embeds
                
            if isinstance(clip_context, list):
                current_clip_context = clip_context[min(current_shot_idx, len(clip_context) - 1)]
            else:
                current_clip_context = clip_context

            sample = pipeline(
                None, 
                num_frames = target_length,
                negative_prompt = None,
                audio_embeds = partial_audio_embeds,
                audio_scale=audio_scale,
                ip_mask = None,
                use_un_ip_mask=use_un_ip_mask,
                height      = sample_height,
                width       = sample_width,
                generator   = generator,
                neg_scale = neg_scale,
                neg_steps = neg_steps,
                use_dynamic_cfg=use_dynamic_cfg,
                use_dynamic_acfg=use_dynamic_acfg,
                guidance_scale = guidance_scale,
                audio_guidance_scale = audio_guidance_scale,
                num_inference_steps = num_inference_steps,
                video      = _input_video,
                mask_video   = _input_video_mask,
                clip_image = None,
                cfg_skip_ratio = cfg_skip_ratio,
                shift = shift,
                clip_context = current_clip_context,
                prompt_embeds = current_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                block_offload=block_offload,
                callback_on_step_end=step_callback,
            ).videos
            
            if save_parts_to_disk:
                part_video_path = os.path.join(folder_paths.output_directory, f"{audio_file_prefix}_part{part_idx:03d}.mp4")
                save_videos_grid(sample, part_video_path, fps=fps)
                part_idx += 1

            if is_new_shot and init_frames != 0:
                new_sample = torch.cat([new_sample, sample], dim=2)
            elif init_frames != 0:
                mix_ratio = torch.from_numpy(
                    np.array([float(i) / float(current_overlap) for i in range(current_overlap)], np.float32)
                ).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(sample.device)
                new_sample[:, :, -current_overlap:] = (
                    new_sample[:, :, -current_overlap:] * (1 - mix_ratio) +
                    sample[:, :, :current_overlap] * mix_ratio
                )
                new_sample = torch.cat([new_sample, sample[:, :, current_overlap:]], dim=2)
            else:
                new_sample = sample

            init_frames += target_length - current_overlap

        # Save merged temporary video
        tmp_video_path = os.path.join(folder_paths.output_directory, f"{audio_file_prefix}_tmp.mp4")
        pli_list=save_videos_grid(new_sample[:,:,:video_length_actual], tmp_video_path, fps=fps)
        
    return pli_list

def Flash_Echo_v3_predata(clip_image_encoder,text_encoder,tokenizer,prompt,negative_prompt,ref_image,sample_size,
                          audio_path,weigths_current_path,fps,video_length,temporal_compression_ratio,device,weight_dtype):
    wav2vec_model_dir=os.path.join(weigths_current_path,"chinese-wav2vec2-base")
    # Load audio
    audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec_model_dir, local_files_only=True).to('cpu')
    audio_encoder.feature_extractor._freeze_parameters()
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_model_dir, local_files_only=True)
   
    audio_clip = AudioFileClip(audio_path)
    video_length_actual = min(int(audio_clip.duration * fps), video_length)
    video_length_actual = int((video_length_actual - 1) // temporal_compression_ratio * temporal_compression_ratio) + 1 if video_length_actual != 1 else 1

    # Get audio features
    mel_input, sr = librosa.load(audio_path, sr=16000) #TODO: 
    mel_input = loudness_norm(mel_input, sr)
    mel_input = mel_input[:int(video_length_actual / 25 * sr)]
    
    print(f"Audio length: {int(len(mel_input)/ sr * 25)}, Video length: {video_length_actual}")
    audio_feature_wav2vec = get_audio_embed(mel_input, wav2vec_feature_extractor, audio_encoder, video_length_actual, sr=16000, fps=25, device='cpu')

    # Get audio batch 
    audio_embeds = audio_feature_wav2vec.to(device=device, dtype=weight_dtype)
    
    indices = (torch.arange(2 * 2 + 1) - 2) * 1 
    center_indices = torch.arange(
        0,  
        video_length_actual,
        1,).unsqueeze(1) + indices.unsqueeze(0)
    center_indices = torch.clamp(center_indices, min=0, max=audio_embeds.shape[0]-1)
    audio_embeds = audio_embeds[center_indices] # F w s c [F, 5, 12, 768]
    audio_embeds = audio_embeds.unsqueeze(0).to(device=device)
    print(f"Audio embeds shape: {audio_embeds.shape}") #Audio embeds shape: torch.Size([1, 97, 5, 12, 768])

    # Load reference image
    source_images = ref_image if isinstance(ref_image, list) else [ref_image]
    num_shots = len(source_images)
    
    ref_start = np.array(source_images[0])
    validation_image_start = Image.fromarray(ref_start).convert("RGB")

    frames_per_shot = video_length_actual // num_shots
    shot_boundaries = [i * frames_per_shot for i in range(num_shots)]
    shot_boundaries.append(video_length_actual)

    if num_shots > 1:
        try:
            intervals = librosa.effects.split(mel_input, top_db=30)
            silence_frames = []
            for i in range(len(intervals) - 1):
                silence_start = intervals[i][1]
                silence_end = intervals[i+1][0]
                silence_mid = (silence_start + silence_end) / 2
                frame = int((silence_mid / sr) * 25) # using fps=25
                silence_frames.append(frame)
            
            for i in range(1, num_shots):
                ideal_b = shot_boundaries[i]
                closest_silence = None
                min_dist = 25 # +/- 1 second window
                for sf in silence_frames:
                    if abs(sf - ideal_b) < min_dist:
                        min_dist = abs(sf - ideal_b)
                        closest_silence = sf
                if closest_silence is not None:
                    shot_boundaries[i] = closest_silence
                    print(f"EchoMimic: Snapped shot boundary {i} to frame {closest_silence} (was {ideal_b})")
        except Exception as e:
            print(f"EchoMimic: Failed to snap shot boundaries: {e}")

    validation_image_end = None
    latent_frames = (video_length_actual - 1) // temporal_compression_ratio + 1
    sample_height, sample_width = get_sample_size(validation_image_start, sample_size)


    input_video, input_video_mask, clip_image = get_image_to_video_latent2(validation_image_start, 
                                                                           validation_image_end, video_length=video_length_actual, sample_size=[sample_height, sample_width])
    # get clip image

    # video_length = init_frames + partial_video_length

    clip_context_list = []
    for img in source_images:
        val_img = Image.fromarray(np.array(img)).convert("RGB")
        _, _, c_img = get_image_to_video_latent2(val_img, None, video_length=1, sample_size=[sample_height, sample_width])
        if c_img is not None:
            c_img = TF.to_tensor(c_img).sub_(0.5).div_(0.5).to(device)
            c_img = c_img.permute(1, 2, 0).unsqueeze(0)
            c_ctx = clip_image_encoder.encode_image(c_img)["penultimate_hidden_states"].to(device, weight_dtype)
        else:
            c_img = Image.new("RGB", (512, 512), color=(0, 0, 0))  
            c_img = TF.to_tensor(c_img).sub_(0.5).div_(0.5).to(device) 
            c_img = c_img.permute(1, 2, 0).unsqueeze(0)
            c_ctx = clip_image_encoder.encode_image(c_img)["penultimate_hidden_states"].to(device, weight_dtype)
            c_ctx = torch.zeros_like(c_ctx)
        clip_context_list.append(c_ctx)
    clip_context = clip_context_list if len(clip_context_list) > 1 else clip_context_list[0]
    clear_comfyui_cache()
    gc.collect()

    prompts = prompt.split("|") if isinstance(prompt, str) else [prompt]
    prompt_embeds_list = []
    for p in prompts:
        pe, ne = encode_prompt(text_encoder,tokenizer,p.strip(),negative_prompt,True,1,device=device,dtype=weight_dtype)
        prompt_embeds_list.append(pe)
        negative_prompt_embeds = ne # Just keep the last negative prompt
    
    prompt_embeds = prompt_embeds_list if len(prompt_embeds_list) > 1 else prompt_embeds_list[0]
    clear_comfyui_cache()
    

    emb={"audio_embeds":audio_embeds,"video_length":video_length,"clip_context":clip_context,"sample_height":sample_height,"sample_width":sample_width,
         "video_length_actual":video_length_actual,"input_video":input_video,"input_video_mask":input_video_mask,
         "prompt_embeds":prompt_embeds,"negative_prompt_embeds":negative_prompt_embeds,
          "ref_image_pil":ref_image,"latent_frames":latent_frames,"shot_boundaries":shot_boundaries,}
    return emb



# def main():
#     args = parse_args()
    
#     # Assign args to original variables
#     config_path = args.config_path
#     model_name = args.model_name
#     ckpt_idx = args.ckpt_idx
#     transformer_path = args.transformer_path
#     vae_path = args.vae_path
#     lora_path = args.lora_path
#     save_path = args.save_path
#     wav2vec_model_dir = args.wav2vec_model_dir
#     image_path = args.image_path
#     audio_path = args.audio_path
#     prompt = args.prompt
#     sampler_name = args.sampler_name
#     video_length = args.video_length
#     guidance_scale = args.guidance_scale
#     audio_guidance_scale = args.audio_guidance_scale
#     audio_scale = args.audio_scale
#     neg_scale = args.neg_scale
#     neg_steps = args.neg_steps
#     num_inference_steps = args.num_inference_steps
#     seed = args.seed
#     lora_weight = args.lora_weight
#     enable_teacache = args.enable_teacache
#     teacache_threshold = args.teacache_threshold
#     num_skip_start_steps = args.num_skip_start_steps
#     teacache_offload = args.teacache_offload
#     use_dynamic_cfg = args.use_dynamic_cfg
#     use_dynamic_acfg = args.use_dynamic_acfg
#     enable_riflex = args.enable_riflex
#     riflex_k = args.riflex_k
#     use_un_ip_mask = args.use_un_ip_mask
#     GPU_memory_mode = args.GPU_memory_mode
#     ulysses_degree = args.ulysses_degree
#     ring_degree = args.ring_degree
#     fsdp_dit = args.fsdp_dit
#     weight_dtype_str = args.weight_dtype
#     sample_size = args.sample_size
#     fps = args.fps
#     add_prompt = args.add_prompt
#     negative_prompt = args.negative_prompt
#     mouth_prompts = args.mouth_prompts
#     cfg_skip_ratio = args.cfg_skip_ratio
#     shift = args.shift
    
#     # Convert weight dtype
#     weight_dtype = torch.bfloat16 if weight_dtype_str == "bfloat16" else torch.float16

#     # Load audio models
#     audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec_model_dir, local_files_only=True).to('cpu')
#     audio_encoder.feature_extractor._freeze_parameters()
#     wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_model_dir, local_files_only=True)

#     device = set_multi_gpus_devices(ulysses_degree, ring_degree)
#     config = OmegaConf.load(config_path)

#     transformer = WanTransformerAudioMask3DModel.from_pretrained(
#         os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
#         transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
#         low_cpu_mem_usage=True if not fsdp_dit else False,
#         torch_dtype=weight_dtype,
#     )
    
#     if transformer_path is not None:
        
#         print(f"From checkpoint: {transformer_path}")
#         if transformer_path.endswith("safetensors"):
#             from safetensors.torch import load_file, safe_open
#             state_dict = load_file(transformer_path)
#         else:
#             state_dict = torch.load(os.path.join(transformer_path, f'checkpoint-{ckpt_idx}.pth'))
#         state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

#         m, u = transformer.load_state_dict(state_dict, strict=False)
#         print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

#     # Get Vae
#     vae = AutoencoderKLWan.from_pretrained(
#         os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
#         additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
#     ).to(weight_dtype)

#     if vae_path is not None:
#         print(f"From checkpoint: {vae_path}")
#         if vae_path.endswith("safetensors"):
#             from safetensors.torch import load_file, safe_open
#             state_dict = load_file(vae_path)
#         else:
#             state_dict = torch.load(vae_path, map_location="cpu")
#         state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

#         m, u = vae.load_state_dict(state_dict, strict=False)
#         print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

#     # Get Tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(
#         os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
#     )

#     # Get Text encoder
#     text_encoder = WanT5EncoderModel.from_pretrained(
#         os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
#         additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
#         low_cpu_mem_usage=True,
#         torch_dtype=weight_dtype,
#     )
#     text_encoder = text_encoder.eval()

#     # Get Clip Image Encoder
#     clip_image_encoder = CLIPModel.from_pretrained(
#         os.path.join(model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
#     ).to(weight_dtype)
#     clip_image_encoder = clip_image_encoder.eval()

#     # Get Scheduler
#     Choosen_Scheduler = scheduler_dict = {
#         "Flow": FlowMatchEulerDiscreteScheduler,
#         "Flow_Unipc": FlowUniPCMultistepScheduler,
#         "Flow_DPM++": FlowDPMSolverMultistepScheduler,
#     }[sampler_name]
#     if sampler_name == "Flow_Unipc" or sampler_name == "Flow_DPM++":
#         config['scheduler_kwargs']['shift'] = 1
#     scheduler = Choosen_Scheduler(
#         **filter_kwargs(Choosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
#     )

#     # Get Pipeline
#     pipeline = WanFunInpaintAudioPipeline(
#         transformer=transformer,
#         vae=vae,
#         tokenizer=tokenizer,
#         text_encoder=text_encoder,
#         scheduler=scheduler,
#         clip_image_encoder=clip_image_encoder
#     )

#     if ulysses_degree > 1 or ring_degree > 1:
#         from functools import partial
#         transformer.enable_multi_gpus_inference()
#         if fsdp_dit:
#             shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
#             pipeline.transformer = shard_fn(pipeline.transformer)


#     pipeline.to(device=device)

#     coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
#     if coefficients is not None:
#         print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
#         pipeline.transformer.enable_teacache(
#             coefficients, num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
#         )

#     generator = torch.Generator(device=device).manual_seed(seed)

#     if lora_path is not None:
#         pipeline = merge_lora(pipeline, lora_path, lora_weight)

#     pipeline.to(device=device)

#     # Create output directory
#     if not os.path.exists(save_path):
#         os.makedirs(save_path, exist_ok=True)

#     with torch.no_grad():
#         # Process single image and audio
#         print(f"Processing: {image_path}")
#         print(f"Audio: {audio_path}")
#         print(f"Prompt: {prompt}")
        
#         # Generate output filename
#         image_name = os.path.basename(image_path).split('.')[0]
#         output_video_path = os.path.join(save_path, f"{image_name}_output.mp4")
        
#         # Check if output already exists
#         if os.path.exists(output_video_path):
#             print(f"⏭️  Skip: {output_video_path} already exists.")
#             return

#         # Load reference image
#         ref_image = Image.open(image_path).convert("RGB")
#         ref_start = np.array(ref_image)

#         # Load audio
#         audio_clip = AudioFileClip(audio_path)
#         video_length_actual = min(int(audio_clip.duration * fps), video_length)
#         video_length_actual = int((video_length_actual - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length_actual != 1 else 1

#         # Get audio features
#         mel_input, sr = librosa.load(audio_path, sr=16000) #TODO: 
#         mel_input = loudness_norm(mel_input, sr)
#         mel_input = mel_input[:int(video_length_actual / 25 * sr)]
        
#         print(f"Audio length: {int(len(mel_input)/ sr * 25)}, Video length: {video_length_actual}")
#         audio_feature_wav2vec = get_audio_embed(mel_input, wav2vec_feature_extractor, audio_encoder, video_length_actual, sr=16000, fps=25, device='cpu')
        
#         # Get audio batch 
#         audio_embeds = audio_feature_wav2vec.to(device=device, dtype=weight_dtype)
        
#         indices = (torch.arange(2 * 2 + 1) - 2) * 1 
#         center_indices = torch.arange(
#             0,  
#             video_length_actual,
#             1,).unsqueeze(1) + indices.unsqueeze(0)
#         center_indices = torch.clamp(center_indices, min=0, max=audio_embeds.shape[0]-1)
#         audio_embeds = audio_embeds[center_indices] # F w s c [F, 5, 12, 768]
#         audio_embeds = audio_embeds.unsqueeze(0).to(device=device)

#         print(f"Audio embeds shape: {audio_embeds.shape}")

#         validation_image_start = Image.fromarray(ref_start).convert("RGB")
#         validation_image_end = None
#         latent_frames = (video_length_actual - 1) // vae.config.temporal_compression_ratio + 1

#         if enable_riflex:
#             pipeline.transformer.enable_riflex(k = riflex_k, L_test = latent_frames)
#         sample_size_0, sample_size_1 = get_sample_size(validation_image_start, sample_size)

#         input_video, input_video_mask, clip_image = get_image_to_video_latent2(validation_image_start, validation_image_end, video_length=video_length_actual, sample_size=[sample_size_0, sample_size_1])

#         sample = pipeline(
#             prompt, 
#             num_frames = video_length_actual,
#             negative_prompt = negative_prompt,
#             audio_embeds = audio_embeds,
#             audio_scale=audio_scale,
#             ip_mask = None,
#             use_un_ip_mask=use_un_ip_mask,
#             height      = sample_size_0,
#             width       = sample_size_1,
#             generator   = generator,
#             neg_scale = neg_scale,
#             neg_steps = neg_steps,
#             use_dynamic_cfg=use_dynamic_cfg,
#             use_dynamic_acfg=use_dynamic_acfg,
#             guidance_scale = guidance_scale,
#             audio_guidance_scale = audio_guidance_scale,
#             num_inference_steps = num_inference_steps,
#             video      = input_video,
#             mask_video   = input_video_mask,
#             clip_image = clip_image,
#             cfg_skip_ratio = cfg_skip_ratio,
#             shift = shift,
#         ).videos

#         # Save temporary video
#         tmp_video_path = os.path.join(save_path, f"{image_name}_tmp.mp4")
#         save_videos_grid(sample[:,:,:video_length_actual], tmp_video_path, fps=fps)
        
#         # Add audio to video
#         video_clip = VideoFileClip(tmp_video_path)
#         audio_clip = audio_clip.subclipped(0, video_length_actual / fps)
#         video_clip = video_clip.with_audio(audio_clip)
#         video_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac", threads=2)

#         # Clean up temporary file
#         os.remove(tmp_video_path)
#         print(f"✅ Saved output to: {output_video_path}")

# if __name__ == "__main__":
#     main()
