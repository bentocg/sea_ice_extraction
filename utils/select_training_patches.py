import os
from PIL import Image
import matplotlib.pyplot as plt
import shutil
import inquirer
from utils.tile_raster import tile_raster
from utils.extract_sea_ice import extract_sea_ice

# globals
tiles_dir = 'tiles'
rasters_dir = '/home/bento/testing_scenes/'
ts_dir = 'training_set_sea_ice'
patch_size = 1200
stride = 1
to_keep = []
to_negative = []

# create training set dir
for ele in ['x', 'y']:
    os.makedirs(f"{ts_dir}/{ele}", exist_ok=True)

# tile and go through patches to populate training set
for scn in os.listdir(rasters_dir):
    if scn.endswith('.tif'):
        tile_raster(f"{rasters_dir}/{scn}", patch_size=patch_size, stride=stride, mask_func=extract_sea_ice, output_dir=tiles_dir)

    # plot tiles and mask side by side
    all_negative = False
    skip_scene = False
    for file in os.listdir(f"{tiles_dir}/x"):
        if skip_scene:
            break

        patch = Image.open(file)
        mask = Image.open(file.replace('/x/', '/y/'))
        figure, axes = plt.subplots(1, 2)
        axes[0, 0].imshow(patch)
        axes[0, 1].imshow(mask)
        plt.show(figure)

        # prompt for choices
        questions = [inquirer.List('keep', message="What do you wish do with this patch?",
                                  choices=['Keep', 'Discard', 'Make it negative', 'Skip scene', 'Make scene negative'])]

        if all_negative:
            to_negative.append(patch)
        answer = inquirer.prompt(questions)


        if answer == 'Skip scene':
            skip_scene = True
            break

        e

        elif answer == 'Discard':
            continue

        elif answer == 'Make it negative':
            to_negative.append()

        elif answer == 'Keep':
            to_keep.append((patch, mask))






    # copy selected images

    # clear tiles
    shutil.rmtree(tiles_dir)



