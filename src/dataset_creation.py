from ffcv_pl.ffcv_utils.generate_dataset import create_image_label_dataset

if __name__ == '__main__':

    # write dataset in ".beton" format
    train_folder = '/media/dserez/datasets/cub/train/'
    test_folder = '/media/dserez/datasets/cub/test/'
    create_image_label_dataset(train_folder=train_folder, test_folder=test_folder)
