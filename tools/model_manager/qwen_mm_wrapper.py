import os
from typing import List, Union

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq


class QwenMultimodalWrapper:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

    def _to_pil(self, img: Union[str, Image.Image]) -> Image.Image:
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        return Image.open(img).convert("RGB")

    def chat_mm(self, texts: List[str], images: List[List[Union[str, Image.Image]]], sampling_params):
        """
        Multimodal chat interface.

        Args:
            texts: List of prompt strings (batched).
            images: List of list-of-images (each inner list corresponds to one prompt).
            sampling_params: List of SamplingParams objects (batched); we use the first one.

        Returns:
            List of generated text responses (one per prompt).
        """
        if isinstance(sampling_params, list):
            sp = sampling_params[0]
        else:
            sp = sampling_params

        outputs: List[str] = []
        for prompt, img_list in zip(texts, images):
            pil_images = [self._to_pil(i) for i in (img_list or [])]
            if pil_images:
                inputs = self.processor(text=prompt, images=pil_images, return_tensors="pt").to(self.device)
            else:
                inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=getattr(sp, "max_tokens", 512),
                do_sample=getattr(sp, "temperature", 0) > 0,
                temperature=getattr(sp, "temperature", 0.0),
                top_p=getattr(sp, "top_p", 1.0),
            )
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            outputs.append(text)

        return outputs

