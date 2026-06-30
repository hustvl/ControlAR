from cleanfid import fid
import argparse

def main(args):
    real_data_path = args.val_images
    gen_data_path = args.generated_images
    cleanfid_score = fid.compute_fid(gen_data_path, real_data_path)
    print(f"The Clean-FID score is {cleanfid_score}")
# /data/vjuicefs_ai_camera_album_ql/public_data/11171709/data/MultiGen20M/val/image
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-images", type=str, required=True)
    parser.add_argument("--generated-images", type=str, required=True)
    args = parser.parse_args()
    main(args)