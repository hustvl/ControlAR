import time
import torch
import numpy as np
from PIL import Image
from safetensors.torch import load_file

from language.t5 import T5Embedder
from condition.canny import CannyDetector
from condition.midas.depth import MidasDetector
from autoregressive.models.generate import generate
from autoregressive.models.gpt_t2i import GPT_models
from tokenizer.tokenizer_image.vq_model import VQ_models


models = {
    "canny": "checkpoints/canny_MR.safetensors",
    "depth": "checkpoints/depth_MR.safetensors",
}


def resize_image_to_16_multiple(image, condition_type='canny'):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    # image = Image.open(image_path)
    width, height = image.size

    if condition_type == 'depth':  # The depth model requires a side length that is a multiple of 32
        new_width = (width + 31) // 32 * 32
        new_height = (height + 31) // 32 * 32
    else:
        new_width = (width + 15) // 16 * 16
        new_height = (height + 15) // 16 * 16

    resized_image = image.resize((new_width, new_height))
    return resized_image


class Model:

    def __init__(self):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.base_model_id = ""
        self.task_name = ""
        self.vq_model = self.load_vq()
        self.t5_model = self.load_t5()
        self.gpt_model_canny = self.load_gpt(condition_type='canny')
        self.gpt_model_depth = self.load_gpt(condition_type='depth')
        self.get_control_canny = CannyDetector()
        self.get_control_depth = MidasDetector(device=self.device)

    def load_vq(self):
        vq_model = VQ_models["VQ-16"](codebook_size=16384,
                                      codebook_embed_dim=8)
        vq_model.to(self.device)
        vq_model.eval()
        checkpoint = torch.load(f"checkpoints/vq_ds16_t2i.pt",
                                map_location="cpu")
        vq_model.load_state_dict(checkpoint["model"])
        del checkpoint
        print("image tokenizer is loaded")
        return vq_model

    def load_gpt(self, condition_type='canny'):
        gpt_ckpt = models[condition_type]
        precision = torch.bfloat16
        latent_size = 768 // 16
        gpt_model = GPT_models["GPT-XL"](
            block_size=latent_size**2,
            cls_token_num=120,
            model_type='t2i',
            condition_type=condition_type,
        ).to(device=self.device, dtype=precision)

        model_weight = load_file(gpt_ckpt)
        gpt_model.load_state_dict(model_weight, strict=False)
        gpt_model.eval()
        print("gpt model is loaded")
        return gpt_model

    def load_t5(self):
        precision = torch.bfloat16
        t5_model = T5Embedder(
            device=self.device,
            local_cache=False,
            cache_dir='checkpoints/flan-t5-xl',
            dir_or_name='flan-t5-xl',
            torch_dtype=precision,
            model_max_length=120,
        )
        return t5_model

    @torch.no_grad()
    def process_canny(
        self,
        image: np.ndarray,
        prompt: str,
        cfg_scale: float,
        temperature: float,
        top_k: int,
        top_p: int,
        seed: int,
        low_threshold: int,
        high_threshold: int,
    ) -> list[Image.Image]:

        image = resize_image_to_16_multiple(image, 'canny')
        W, H = image.size
        print(W, H)
        condition_img = self.get_control_canny(np.array(image), low_threshold,
                                               high_threshold)
        condition_img = torch.from_numpy(condition_img[None, None,
                                                       ...]).repeat(
                                                           2, 3, 1, 1)
        condition_img = condition_img.to(self.device)
        condition_img = 2 * (condition_img / 255 - 0.5)
        prompts = [prompt] * 2
        caption_embs, emb_masks = self.t5_model.get_text_embeddings(prompts)

        print(f"processing left-padding...")
        new_emb_masks = torch.flip(emb_masks, dims=[-1])
        new_caption_embs = []
        for idx, (caption_emb,
                  emb_mask) in enumerate(zip(caption_embs, emb_masks)):
            valid_num = int(emb_mask.sum().item())
            print(f'  prompt {idx} token len: {valid_num}')
            new_caption_emb = torch.cat(
                [caption_emb[valid_num:], caption_emb[:valid_num]])
            new_caption_embs.append(new_caption_emb)
        new_caption_embs = torch.stack(new_caption_embs)
        c_indices = new_caption_embs * new_emb_masks[:, :, None]
        c_emb_masks = new_emb_masks
        qzshape = [len(c_indices), 8, H // 16, W // 16]
        t1 = time.time()
        index_sample = generate(
            self.gpt_model_canny,
            c_indices,
            (H // 16) * (W // 16),
            c_emb_masks,
            condition=condition_img,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            sample_logits=True,
        )
        sampling_time = time.time() - t1
        print(f"Full sampling takes about {sampling_time:.2f} seconds.")

        t2 = time.time()
        print(index_sample.shape)
        samples = self.vq_model.decode_code(
            index_sample, qzshape)  # output value is between [-1, 1]
        decoder_time = time.time() - t2
        print(f"decoder takes about {decoder_time:.2f} seconds.")

        samples = torch.cat((condition_img[0:1], samples), dim=0)
        samples = 255 * (samples * 0.5 + 0.5)
        samples = [image] + [
            Image.fromarray(
                sample.permute(1, 2, 0).cpu().detach().numpy().clip(
                    0, 255).astype(np.uint8)) for sample in samples
        ]
        del condition_img
        torch.cuda.empty_cache()
        return samples

    @torch.no_grad()
    def process_depth(
        self,
        image: np.ndarray,
        prompt: str,
        cfg_scale: float,
        temperature: float,
        top_k: int,
        top_p: int,
        seed: int,
    ) -> list[Image.Image]:
        image = resize_image_to_16_multiple(image, 'depth')
        W, H = image.size
        print(W, H)
        image_tensor = torch.from_numpy(np.array(image)).to(self.device)
        condition_img = torch.from_numpy(
            self.get_control_depth(image_tensor)).unsqueeze(0)
        condition_img = condition_img.unsqueeze(0).repeat(2, 3, 1, 1)
        condition_img = condition_img.to(self.device)
        condition_img = 2 * (condition_img / 255 - 0.5)
        prompts = [prompt] * 2
        caption_embs, emb_masks = self.t5_model.get_text_embeddings(prompts)

        print(f"processing left-padding...")
        new_emb_masks = torch.flip(emb_masks, dims=[-1])
        new_caption_embs = []
        for idx, (caption_emb,
                  emb_mask) in enumerate(zip(caption_embs, emb_masks)):
            valid_num = int(emb_mask.sum().item())
            print(f'  prompt {idx} token len: {valid_num}')
            new_caption_emb = torch.cat(
                [caption_emb[valid_num:], caption_emb[:valid_num]])
            new_caption_embs.append(new_caption_emb)
        new_caption_embs = torch.stack(new_caption_embs)

        c_indices = new_caption_embs * new_emb_masks[:, :, None]
        c_emb_masks = new_emb_masks
        qzshape = [len(c_indices), 8, H // 16, W // 16]
        t1 = time.time()
        index_sample = generate(
            self.gpt_model_depth,
            c_indices,
            (H // 16) * (W // 16),
            c_emb_masks,
            condition=condition_img,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            sample_logits=True,
        )
        sampling_time = time.time() - t1
        print(f"Full sampling takes about {sampling_time:.2f} seconds.")

        t2 = time.time()
        print(index_sample.shape)
        samples = self.vq_model.decode_code(index_sample, qzshape)
        decoder_time = time.time() - t2
        print(f"decoder takes about {decoder_time:.2f} seconds.")
        condition_img = condition_img.cpu()
        samples = samples.cpu()
        samples = torch.cat((condition_img[0:1], samples), dim=0)
        samples = 255 * (samples * 0.5 + 0.5)
        samples = [image] + [
            Image.fromarray(
                sample.permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8))
            for sample in samples
        ]
        del image_tensor
        del condition_img
        torch.cuda.empty_cache()
        return samples
