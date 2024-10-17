import glob
import os
import argparse

import torch
from tqdm import tqdm
from plip import PLIP
import numpy as np
from PIL import Image


def arg_parse():
    parser = argparse.ArgumentParser(description='PLIP for TCGA feature extraction arguments.')

    parser.add_argument('--dataset', type=str,
                        default='path_of_dataset_id_file/TCGA-BRCA.txt',
                        help='Path of dataset txt file path.')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size.')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='Device for training.')
    parser.add_argument('--output-path', default='path_to_save_feature/TCGA-BRCA/', type=str,
                        help='Path of output feature.')

    return parser.parse_args()


def main(args):
    plip = PLIP(model_name='vinid/plip', device=args.device)
    dataset = args.dataset
    BATCH_SIZE = args.batch_size

    with open(dataset, 'r') as f:
        slides_filepaths = f.read().splitlines()

    for slide_path in tqdm(slides_filepaths):
        slide_id = os.path.basename(slide_path)

        feature_name = os.path.join(args.output_path, slide_id + '.pth')

        if os.path.exists(feature_name):
            continue

        patch_path = os.path.join(slide_path, 'Medium')
        patch_list = glob.glob(patch_path + '/*.jpg')
        patch_coords_list = []

        for patch in patch_list:
            patch_coords = patch.split('.jpg')[0]
            patch_coords = os.path.basename(patch_coords)
            patch_coords_list.append(patch_coords)

        all_patch_list = [patch_list[i:i + BATCH_SIZE] for i in range(0, len(patch_list), BATCH_SIZE)]
        feature = {}
        slide_feature = None

        for item_patch_list in all_patch_list:
            images = []
            for patch_path in item_patch_list:
                images.append(Image.open(patch_path))
            image_embeddings = plip.encode_images(images, batch_size=BATCH_SIZE)

            if slide_feature is None:
                slide_feature = image_embeddings
            else:
                slide_feature = np.concatenate((slide_feature, image_embeddings), axis=0)

        feature['feature'] = slide_feature
        feature['coords'] = patch_coords_list

        torch.save(feature, feature_name)


if __name__ == "__main__":
    args = arg_parse()
    main(args)
