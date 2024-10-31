from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import argparse


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for file in tqdm(os.listdir(sample_dir), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{file}")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-images", type=str, required=True)
    args = parser.parse_args()    
    num_fid_samples = 50000
    create_npz_from_sample_folder(args.generated_images, num_fid_samples)
    print("Done.")