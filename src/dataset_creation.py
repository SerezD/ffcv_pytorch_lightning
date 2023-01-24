from ffcv_pl.ffcv_utils.generate_dataset import create_image_label_dataset

if __name__ == '__main__':

    # write dataset in ".beton" format
    test_folder = '/media/dserez/datasets/imagenet/test/'
    create_image_label_dataset(test_folder=test_folder)
