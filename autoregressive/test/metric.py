import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import f1_score
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.classification import BinaryF1Score

class SSIM:
    def __init__(self, data_range=1.0):
        ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)#.to(device)
        self.total_ssim = 0.0
        self.count = 0
    def update(self, img1, img2):
        ssim_value = ssim((img1/255).clip(0,1), (img2/255).clip(0,1), data_range=1.0)
        self.total_ssim += ssim_value
        self.count += 1

    def calculate(self):
        if self.count == 0:
            raise ValueError("No images have been added.")
        return self.total_ssim / self.count   
        
        
        
class F1score:
    def __init__(self, threshold=128):
        self.threshold = threshold
        self.total_f1 = 0
        self.count = 0

    def update(self, img1, img2):
        
        assert img1.size == img2.size, "The images must be the same size."
    
        binary_image1 = (img1 > self.threshold).astype(int)
        binary_image2 = (img2 > self.threshold).astype(int)

        y_true = binary_image1.flatten()
        y_pred = binary_image2.flatten()

        f1 = f1_score(y_true, y_pred)

        self.total_f1 += f1
        self.count += 1

    def calculate(self):
        average_f1 = self.total_f1 / self.count
        return average_f1

class RMSE:
    def __init__(self):
        self.total_rmse = 0
        self.count = 0

    def update(self, img1, img2):
        
        assert img1.size == img2.size, "The images must be the same size."
        diff = img1 - img2
        diff_squared = np.square(diff)
        mse = np.mean(diff_squared)
        rmse = np.sqrt(mse)
        self.total_rmse += rmse
        self.count += 1

    def calculate(self):
        average_f1 = self.total_rmse / self.count
        return average_f1





if __name__ == "__main__":
    img1_1 = np.random.randn(256,256)
    img1_1 = img1_1 - img1_1.min()
    img1_1 = 255*img1_1/img1_1.max()
    img1_1 = img1_1.astype(np.uint8)
    img1_2 = np.random.randn(256,256)
    img1_2 = img1_2 - img1_2.min()
    img1_2 = 255*img1_2/img1_2.max()
    img1_2 = img1_2.astype(np.uint8)
    img2_1 = np.random.randn(256,256)
    img2_1 = img2_1 - img2_1.min()
    img2_1 = 255*img2_1/img2_1.max()
    img2_1 = img2_1.astype(np.uint8)
    img2_2 = np.random.randn(256,256)
    img2_2 = img2_2 - img2_2.min()
    img2_2 = 255*img2_2/img2_2.max()
    img2_2 = img2_2.astype(np.uint8)
    img_pairs = [(img1_1, img2_1), (img1_2, img2_2)]
    
    calculator = AverageSSIMCalculator()
    
    for img1, img2 in img_pairs:
        calculator.add_images(img1, img2)
    
    avg_ssim = calculator.calculate_average_ssim()
    print(f'Average SSIM: {avg_ssim}')