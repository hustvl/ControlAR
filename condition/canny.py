import cv2
import torch
import numpy as np


class CannyDetector:
    def __call__(self, img, low_threshold=100, high_threshold=200):
        """
        input: array or tensor (H,W,3)
        output: array (H,W)
        """
        if torch.is_tensor(img):
            img = img.cpu().detach().numpy().astype(np.uint8)
        return cv2.Canny(img, low_threshold, high_threshold)
    

if __name__ == '__main__':
    apply_canny = CannyDetector()
    img = cv2.imread('condition/dragon_resize.png')
    import numpy as np
    print(img.max())
    detected_map = apply_canny(img, 100, 200)
    print(detected_map.shape, detected_map.max(), detected_map.min())
    cv2.imwrite('condition/example_canny.jpg', detected_map)
    np.save('condition/example_canny.npy', detected_map[None,None])