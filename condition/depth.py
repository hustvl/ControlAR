from controlnet_aux import LineartDetector
import torch
import cv2
import numpy as np
from transformers import DPTImageProcessor, DPTForDepthEstimation
class Depth:
    def __init__(self, device):
        self.model = DPTForDepthEstimation.from_pretrained("condition/ckpts/dpt_large")
        
    def __call__(self, input_image):
        """
        input: tensor()
        """
        control_image = self.model(input_image)
        return np.array(control_image)
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from transformers import DPTImageProcessor, DPTForDepthEstimation
    from PIL import Image

    image = Image.open('condition/example/t2i/depth/depth.png')
    img = cv2.imread('condition/example/t2i/depth/depth.png')
    processor = DPTImageProcessor.from_pretrained("condition/ckpts/dpt_large")
    model = DPTForDepthEstimation.from_pretrained("condition/ckpts/dpt_large")

    inputs = torch.from_numpy(np.array(img)).permute(2,0,1).unsqueeze(0).float()#
    inputs = 2*(inputs/255 - 0.5)
    inputs = processor(images=image, return_tensors="pt", size=(512,512))
    print(inputs)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    print(predicted_depth.shape)
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    
    depth = Image.fromarray(formatted)
    depth.save('condition/example/t2i/depth/example_depth.jpg')