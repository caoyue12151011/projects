'''
To analyze distances of bases in starblast.io.
'''
import cv2
import glob
import pyperclip
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# settings to make matplotlib plots look better
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


def image_to_map(file, map_size, clipboard=False):
    '''
    To turn an image into an asteroid field map.

    Inputs
    ------
    file: str, path of the image file. Image should be (m*n*3) or (m*n) 
          array, 0-255
    map_size: 20-200, even values only
    clipboard: whether to copy map string to clipboard

    Returns
    -------
    map: str
    '''
    name = file.split('/')[-1].split('.')[0]

    # load image
    img = cv2.imread(file)

    # rgb to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # resize img
    img = cv2.resize(img, (map_size, map_size))

    # norm img
    img = (img - img.min()) / (img.max() - img.min())

    # range to 0-9
    img = (10*img).astype('uint8')
    img[img==10] = 9

    # will attempt to draw with lowest asteroid values
    # invert color scale if needed
    if np.mean(img) > 4.5:
        img = 9-img

    # demo the preview image
    plt.figure()
    plt.imshow(img)
    plt.title(f'Asteroid field map of {file}\n')
    plt.tight_layout()
    plt.savefig(f'image/{name}_preview_size{map_size}.pdf')
    plt.close()

    # img to map string
    img = img.astype(str)
    map = r'\n'.join([''.join(row) for row in img])
    map = 'var map = "' + map + '";'

    # save to txt file 
    with open(f'map/{name}_size{map_size}.txt', 'w') as f:
        f.write(map)

    # copy to clipboard
    if clipboard:
        pyperclip.copy(map)
        print(f'Map string of {file} (size={map_size}) copied to '
               'clipboard. '
               'Do not foget add "custom_map: map," in your code.')


# parameters 
map_size = 200

# get all image files 
files = glob.iglob('image/*.*g')
for file in files:
    image_to_map(file, map_size)