import os
import logging
import numpy as np
import nibabel as nib
import blosc2

from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_closing
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.measure import label


def mm_to_vox(mm, spacing):
    """Convert mm to voxel units (rounded, min 1)."""
    return tuple(max(1, int(round(m / s))) for m, s in zip(mm, spacing))


def extract_body_mask_dixon(
    fat,
    water,
    spacing=(2.23, 2.23, 3.0),  # mm
    smooth_mm=(6.0, 6.0, 3.0),
    closing_mm=(16.0, 16.0, 9.0),
    min_size_ratio=0.02,
    clip_percentiles=(1, 99),
):
    """
    Extract conservative body mask from Dixon fat/water MRI.
    """

    assert fat.shape == water.shape
    assert fat.ndim == 3

    fat = fat.astype(np.float32)
    water = water.astype(np.float32)

    # --------------------------------------------------
    # 1. Robust per-channel normalization
    # --------------------------------------------------
    def normalize(x):
        p1, p99 = np.percentile(x, clip_percentiles)
        x = np.clip(x, p1, p99)
        return (x - p1) / (p99 - p1 + 1e-6)

    fat_n = normalize(fat)
    water_n = normalize(water)

    # --------------------------------------------------
    # 2. Combine fat + water (key Dixon trick)
    # --------------------------------------------------
    body_signal = fat_n + water_n

    # --------------------------------------------------
    # 3. Spacing-aware smoothing
    # --------------------------------------------------
    smooth_sigma = mm_to_vox(smooth_mm, spacing)
    body_smooth = gaussian_filter(body_signal, sigma=smooth_sigma)

    # --------------------------------------------------
    # 4. Otsu threshold
    # --------------------------------------------------
    thresh = threshold_otsu(body_smooth)
    mask = body_smooth > thresh

    # --------------------------------------------------
    # 5. Morphological cleanup
    # --------------------------------------------------
    mask = binary_fill_holes(mask)

    min_size = int(min_size_ratio * mask.size)
    mask = remove_small_objects(mask, min_size=min_size)

    closing_kernel = np.ones(mm_to_vox(closing_mm, spacing), dtype=bool)
    mask = binary_closing(mask, structure=closing_kernel)

    # --------------------------------------------------
    # 6. Keep largest connected component
    # --------------------------------------------------
    labeled = label(mask)
    if labeled.max() > 1:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        mask = labeled == sizes.argmax()

    return mask.astype(bool)


def save_blosc2_array(arr, out_path):
    """
    Compress and save numpy array as blosc2 file
    """
    packed = blosc2.pack_array(arr, codec=blosc2.Codec.ZSTD)
    with open(out_path, "wb") as f:
        f.write(packed)


def process_directory(root_dir):

    log_path = os.path.join(root_dir, "body_masks.txt")

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info("Starting processing")

    for dirpath, _, filenames in os.walk(root_dir):
        if "wat.nii.gz" in filenames:
            if "mask.b2nd" in filenames:
                logging.info(f"Skipping already processed: {os.path.basename(dirpath)}")
                continue
            try:
                print(f"Processing: {dirpath}")

                wat_path = os.path.join(dirpath, "wat.nii.gz")
                wat_img = nib.load(wat_path)
                wat_data = wat_img.get_fdata(dtype=np.float32)
                # wat_mask = extract_body_mask_dixon(
                #     wat_data, spacing=wat_img.header.get_zooms()[:3]
                # )

                fat_path = os.path.join(dirpath, "fat.nii.gz")
                fat_data = nib.load(fat_path).get_fdata(dtype=np.float32)
                mask = extract_body_mask_dixon(
                    fat_data, wat_data, spacing=wat_img.header.get_zooms()[:3]
                )

                mask_blosc_path = os.path.join(dirpath, "mask.b2nd")
                save_blosc2_array(mask, mask_blosc_path)

                logging.info(f"Successfully processed: {os.path.basename(dirpath)}")
            except Exception as e:
                logging.error(f"Failed processing {os.path.basename(dirpath)}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert wat/fat NIfTI to Blosc2 + metadata pickle"
    )
    parser.add_argument("folder", help="Root directory to search")

    args = parser.parse_args()

    root = "/path/to/ukbb/dataset/"

    process_directory(os.path.join(root, args.folder))
