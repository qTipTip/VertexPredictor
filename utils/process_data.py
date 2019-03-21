import argparse
import json
import pathlib

import PIL.Image
import matplotlib.patches as plp
import matplotlib.pyplot as plt
import numpy as np
import tqdm


def create_bounding_box_from_polygon(polygon, original_image_size, padding_factor=0.15):
    """
    Given a list of vertices, create a bounding box, with an optional padding in percent. Clipping is performed,
    to make sure the bounding box is within the original image.

    :param original_image_size: [w, h] width and height of image the polygon is taken from
    :param polygon: array like of shape (num_vertices, 2)
    :param padding_factor: number in the interval [0, 1]
    :return: the lower left and upper right corner of the bounding box, as a (4,) array
    """

    polygon = np.array(polygon)
    ll_corner = np.min(polygon, axis=0)
    ur_corner = np.max(polygon, axis=0)
    h_and_w = ur_corner - ll_corner

    pad = np.rint(padding_factor * h_and_w).astype(np.int)
    ll_corner -= pad
    ur_corner += pad

    ll_corner.clip(0, original_image_size, out=ll_corner)
    ur_corner.clip(0, original_image_size, out=ur_corner)

    return np.concatenate((ll_corner, ur_corner), axis=0)


def trivial_area(bounding_box, tol=1.0e-15):
    a, b, c, d = bounding_box
    if abs(c - a) < tol or abs(d - b) < tol:
        return True
    return False


def process_cityscapes_dataset(root_folder='.', data_set='train', output_folder=None, padding_factor=0.15,
                               output_image_size=224):
    """
    This function iterates through root_folder/dataset, opens each image, extracts a cropped image for each annotated
    object in the original image.

    :param root_folder: the root folder of the cityscapes dataset, such that the data is organized as
        root_folder/{train,test,val}
    :param data_set: 'train'/'test'/'val' which set to extract images from.
    :param padding_factor: how much to pad the bounding box
    :param output_folder: desired output folder. If None, uses root_folder/output
    :param output_image_size: the size of the cropped images. Images are resized before saving.
    :return: None
    """
    raw_path_str = '/'.join([root_folder, data_set])
    raw_path = pathlib.Path(raw_path_str)

    if not raw_path.exists():
        raise FileNotFoundError(f'The directory {raw_path.resolve()} was not found.')

    files = list(raw_path.glob('*/*.png'))

    if output_folder is None:
        output_folder = 'output'
    output_folder_path_str = '/'.join([root_folder, output_folder, data_set])
    output_folder_path = pathlib.Path(output_folder_path_str)
    verify_output_directories(output_folder_path)

    suffixes = ['_labelIds.png', '_color.png', '_polygons.json', 'instanceIds.json']

    cropped_image_counter = 0
    for f in tqdm.tqdm(files, desc=f'{"Iterating over images":<50}', leave=False):

        # For each file, we load the json-object containing the polygon-information, and open a PIL.Image
        file_handle = '_'.join(f.name.split('_')[:-1])
        json_file_path = pathlib.Path(str(f.parents[0]) + '/' + file_handle + suffixes[2])
        json_file = json.load(open(json_file_path.resolve()))

        image_f = PIL.Image.open(f)
        image_h = json_file['imgHeight']
        image_w = json_file['imgWidth']

        # For each object in the file, we extract the polygon, compute a bounding box, and crop the image accordingly.
        # Note that we also have to translate and rescale the labeled polygon, as we resize the ground truth image.
        for object in tqdm.tqdm(json_file['objects'], desc=f"{f.name:<50}"):
            polygon = np.array(object['polygon'], dtype=np.float64)

            bounding_box = create_bounding_box_from_polygon(polygon, [image_w, image_h], padding_factor=padding_factor)

            if trivial_area(bounding_box):
                print('Trivial bounding-box with zero area: SKIPPING')
                continue
            resize_crop_and_save_image(bounding_box, image_f, output_image_size, output_folder_path,
                                       cropped_image_counter)
            resize_translate_and_save_polygon(bounding_box, output_image_size, polygon, output_folder_path,
                                              cropped_image_counter)

            cropped_image_counter += 1


def resize_crop_and_save_image(bounding_box, image_f, output_image_size, output_folder_path, cropped_image_counter):
    """
    Resizes, crops and saves the image according to the given bounding box, and the output_image_size

    :param bounding_box:
    :param image_f:
    :param output_image_size:
    :return:
    """
    cropped_image = image_f.crop(box=bounding_box)
    cropped_image_resized = cropped_image.resize((output_image_size, output_image_size), PIL.Image.BILINEAR)
    cropped_image_resized.save(f"{(output_folder_path / 'images').absolute()}/{cropped_image_counter:010d}.png",
                               'PNG')


def resize_translate_and_save_polygon(bounding_box, output_image_size, polygon, output_folder_path,
                                      cropped_image_counter):
    """
    Resizes, translates and saves polygon corresponding to the cropped image.
    :param bounding_box:
    :param output_image_size:
    :param polygon:
    :return:
    """
    bounding_box_w = bounding_box[2] - bounding_box[0]
    bounding_box_h = bounding_box[3] - bounding_box[1]
    polygon_scale_w = output_image_size / bounding_box_w
    polygon_scale_h = output_image_size / bounding_box_h
    polygon[:, 0] -= bounding_box[0]
    polygon[:, 1] -= bounding_box[1]
    polygon[:, 0] *= polygon_scale_w
    polygon[:, 1] *= polygon_scale_h
    polygon[:, 0] = polygon[:, 0].clip(0, np.minimum(output_image_size - 1, polygon[:, 0]))
    polygon[:, 1] = polygon[:, 1].clip(0, np.minimum(output_image_size - 1, polygon[:, 1]))
    polygon = polygon.astype(np.int).tolist()
    with open(f"{(output_folder_path / 'labels').absolute()}/{cropped_image_counter:010d}.json", 'w') as polygon_out:
        json.dump({'polygon': polygon}, polygon_out)


def verify_output_directories(output_folder_path):
    if not (output_folder_path / 'images').is_dir():
        print(f'{(output_folder_path / "images").absolute()} does not exists. This directory will be created')
        (output_folder_path / "images").mkdir(parents=True, exist_ok=True)
    if not (output_folder_path / 'labels').is_dir():
        print(f'{(output_folder_path / "labels").absolute()} does not exists. This directory will be created')
        (output_folder_path / "labels").mkdir(parents=True, exist_ok=True)


def plot_polygon_and_bounding_box(image_f, label='N/A', polygon=None, bounding_box=None):
    """
    Given a PIL image, a polygon and a bounding box, along with the label. Plot it.

    :param bounding_box:
    :param image_f:
    :param label:
    :param polygon:
    :return:
    """
    if polygon is not None:
        plt.scatter(polygon[:, 0], polygon[:, 1])
        plt.plot(polygon[:, 0], polygon[:, 1])

    if bounding_box is not None:
        ax = plt.gca()
        ax.add_patch(
            plp.Rectangle(bounding_box[:2], bounding_box[2] - bounding_box[0], bounding_box[3] - bounding_box[1],
                          alpha=0.3))
    plt.imshow(image_f[0])
    plt.title(label)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CityScapes Data Processer")

    parser.add_argument('--data_set', default='train', type=str, help='train/test/val')
    parser.add_argument('--data_root', default='../gtFine', type=str, help='The path to the gtFine root')
    parser.add_argument('--padding', default=0.15, type=float,
                        help='A factor between 0 and 1 denoting the padding used for the bounding box')
    parser.add_argument('--output_size', default=224, type=int, help='The output resolution of the images')
    args = parser.parse_args()

    process_cityscapes_dataset(root_folder=args.data_root, data_set=args.data_set, padding_factor=args.padding,
                               output_image_size=args.output_size)
