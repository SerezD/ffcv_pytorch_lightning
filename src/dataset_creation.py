from src.ffcv_pl_loader.ffcv_utils.generate_dataset import create_image_dataset

if __name__ == '__main__':

    # write dataset in ".beton" format
    train_folder = '/media/dserez/datasets/cub2002011/test/'
    create_image_dataset(train_folder)

