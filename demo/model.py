import gc
import spaces
from safetensors.torch import load_file
from autoregressive.models.gpt_t2i import GPT_models
from tokenizer.tokenizer_image.vq_model import VQ_models
from language.t5 import T5Embedder
import torch
import numpy as np
import PIL
from PIL import Image
from condition.canny import CannyDetector
import time
from autoregressive.models.generate import generate
from condition.midas.depth import MidasDetector
from preprocessor import Preprocessor

models = {
    "edge": "checkpoints/edge_base.safetensors",
    "depth": "checkpoints/depth_base.safetensors",
}
class Model:
    def __init__(self):
        self.device = torch.device(
            "cuda")
        self.base_model_id = ""
        self.task_name = ""
        self.vq_model = self.load_vq()
        self.t5_model = self.load_t5()
        # self.gpt_model_edge = self.load_gpt(condition_type='edge')
        # self.gpt_model_depth = self.load_gpt(condition_type='depth')
        self.gpt_model = self.load_gpt()
        self.preprocessor = Preprocessor()

    def to(self, device):
        self.gpt_model_canny.to('cuda')

    def load_vq(self):
        vq_model = VQ_models["VQ-16"](codebook_size=16384,
                                      codebook_embed_dim=8)
        vq_model.eval()
        checkpoint = torch.load(f"checkpoints/vq_ds16_t2i.pt",
                                map_location="cpu")
        vq_model.load_state_dict(checkpoint["model"])
        del checkpoint
        print("image tokenizer is loaded")
        return vq_model

    def load_gpt(self, condition_type='edge'):
        # gpt_ckpt = models[condition_type]
        # precision = torch.bfloat16
        precision = torch.float32
        latent_size = 512 // 16
        gpt_model = GPT_models["GPT-XL"](
            block_size=latent_size**2,
            cls_token_num=120,
            model_type='t2i',
            condition_type=condition_type,
            adapter_size='base',
        ).to(device='cpu', dtype=precision)
        # model_weight = load_file(gpt_ckpt)
        # gpt_model.load_state_dict(model_weight, strict=False)
        # gpt_model.eval()
        # print("gpt model is loaded")
        return gpt_model

    def load_gpt_weight(self, condition_type='edge'):
        torch.cuda.empty_cache()
        gc.collect()
        gpt_ckpt = models[condition_type]
        model_weight = load_file(gpt_ckpt)
        self.gpt_model.load_state_dict(model_weight, strict=False)
        self.gpt_model.eval()
        torch.cuda.empty_cache()
        gc.collect()
        # print("gpt model is loaded")
        
    def load_t5(self):
        # precision = torch.bfloat16
        precision = torch.float32
        t5_model = T5Embedder(
            device=self.device,
            local_cache=True,
            cache_dir='checkpoints/flan-t5-xl',
            dir_or_name='flan-t5-xl',
            torch_dtype=precision,
            model_max_length=120,
        )
        return t5_model

    @torch.no_grad()
    @spaces.GPU(enable_queue=True)
    def process_edge(
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
        control_strength: float,
        preprocessor_name: str,
    ) -> list[PIL.Image.Image]:
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        origin_W, origin_H = image.size
        if preprocessor_name == 'Canny':
            self.preprocessor.load("Canny")
            condition_img = self.preprocessor(
                image=image, low_threshold=low_threshold, high_threshold=high_threshold, detect_resolution=512)
        elif preprocessor_name == 'Hed':
            self.preprocessor.load("HED")
            condition_img = self.preprocessor(
                image=image,image_resolution=512, detect_resolution=512)
        elif preprocessor_name == 'Lineart':
            self.preprocessor.load("Lineart")
            condition_img = self.preprocessor(
                image=image,image_resolution=512, detect_resolution=512)
        elif preprocessor_name == 'No preprocess':
            condition_img = image
        print('get edge')
        del self.preprocessor.model
        torch.cuda.empty_cache()
        condition_img = condition_img.resize((512,512))
        W, H = condition_img.size

        self.t5_model.model.to('cuda').to(torch.bfloat16)
        self.load_gpt_weight('edge')
        self.gpt_model.to('cuda').to(torch.bfloat16)
        self.vq_model.to('cuda')
        condition_img = torch.from_numpy(np.array(condition_img)).unsqueeze(0).permute(0,3,1,2).repeat(1,1,1,1)
        condition_img = condition_img.to(self.device)
        condition_img = 2*(condition_img/255 - 0.5)
        prompts = [prompt] * 1
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
        print(caption_embs.device)
        index_sample = generate(
            self.gpt_model,
            c_indices,
            (H // 16) * (W // 16),
            c_emb_masks,
            condition=condition_img,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            sample_logits=True,
            control_strength=control_strength,
        )
        sampling_time = time.time() - t1
        print(f"Full sampling takes about {sampling_time:.2f} seconds.")

        t2 = time.time()
        print(index_sample.shape)
        samples = self.vq_model.decode_code(
            index_sample, qzshape)  # output value is between [-1, 1]
        decoder_time = time.time() - t2
        print(f"decoder takes about {decoder_time:.2f} seconds.")
        # samples = condition_img[0:1]
        samples = torch.cat((condition_img[0:1], samples), dim=0)
        samples = 255 * (samples * 0.5 + 0.5)
        samples = [
            Image.fromarray(
                sample.permute(1, 2, 0).cpu().detach().numpy().clip(
                    0, 255).astype(np.uint8)) for sample in samples
        ]
        del condition_img
        torch.cuda.empty_cache()
        return samples

    @torch.no_grad()
    @spaces.GPU(enable_queue=True)
    def process_depth(
        self,
        image: np.ndarray,
        prompt: str,
        cfg_scale: float,
        temperature: float,
        top_k: int,
        top_p: int,
        seed: int,
        control_strength: float,
        preprocessor_name: str
    ) -> list[PIL.Image.Image]:
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        origin_W, origin_H = image.size
        # print(image)
        if preprocessor_name == 'depth':
            self.preprocessor.load("Depth")
            condition_img = self.preprocessor(
                    image=image,
                    image_resolution=512,
                    detect_resolution=512,
                )
        elif preprocessor_name == 'No preprocess':
            condition_img = image
        print('get depth')
        del self.preprocessor.model
        torch.cuda.empty_cache()
        condition_img = condition_img.resize((512,512))
        W, H = condition_img.size

        self.t5_model.model.to(self.device).to(torch.bfloat16)
        self.load_gpt_weight('depth')
        self.gpt_model.to('cuda').to(torch.bfloat16)
        self.vq_model.to(self.device)
        condition_img = torch.from_numpy(np.array(condition_img)).unsqueeze(0).permute(0,3,1,2).repeat(1,1,1,1)
        condition_img = condition_img.to(self.device)
        condition_img = 2*(condition_img/255 - 0.5)
        prompts = [prompt] * 1
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
            self.gpt_model,
            c_indices,
            (H // 16) * (W // 16),
            c_emb_masks,
            condition=condition_img,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            sample_logits=True,
            control_strength=control_strength,
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

        # samples = condition_img[0:1]
        samples = torch.cat((condition_img[0:1], samples), dim=0)
        samples = 255 * (samples * 0.5 + 0.5)
        samples = [
            Image.fromarray(
                sample.permute(1, 2, 0).cpu().detach().numpy().clip(0, 255).astype(np.uint8))
            for sample in samples
        ]
        del condition_img
        torch.cuda.empty_cache()
        return samples
