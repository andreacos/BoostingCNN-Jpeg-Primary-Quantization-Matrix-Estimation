"""
    2020 Department of Information Engineering and Mathematics, University of Siena, Italy.

    Authors:  Benedetta Tondi (benedettatondi@gmail.com) and Andrea Costanzo (anreacos82@gmail.com)

    This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
    License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
    version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
    implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
    more details. You should have received a copy of the GNU General Public License along with this program.
    If not, see <http://www.gnu.org/licenses/>.
"""


import os
import cv2
import math
import numpy as np
from glob import glob
from skimage.transform import resize
import configuration as cfg
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from skimage.util import view_as_windows
from utils import qf1_qf2_coefficients_map, max_min_coefficient, label2coefficient
from networks import custom_categorical, custom_softmax_activation

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def batch_label2_coefficient(predictions, max_coeffs):

    labels = np.zeros((predictions.shape[0], cfg.max_no_Q_coefs))

    for i in range(0, predictions.shape[0]):
        labels[i] = label2coefficient(predictions[i].flatten(), max_coeffs)

    return labels


def show_coefficients(data, ncoef=15, viewsize=(64, 64)):

    data = data[:, :, :ncoef]
    batch = np.zeros((data.shape[2], data.shape[0], data.shape[1], 3), np.uint8)
    for i in range(0, data.shape[2]):
        plt.imsave('temp.png', data[:, :, 0], cmap='jet')
        layer = plt.imread('temp.png')
        os.remove('temp.png')
        batch[i] = np.uint8(255 * layer[:, :, :3])

    viewsize += (batch.shape[-1],)

    # Display the batch images over a square grid
    N = int(math.ceil(math.sqrt(len(batch))))

    map = np.kron(np.reshape(np.arange(0, N ** 2), (N, N)), np.ones(viewsize[:2], dtype=np.uint8))

    if batch.shape[-1] == 3:
        map = np.repeat(np.expand_dims(map, -1), 3, axis=2)

    batchimage = map.copy()

    # Place images into the grid, missing images to reach grid size are padded with black pixels
    for i in range(0, len(batch)):
        img = batch[i, :, :, :]

        if batch.shape[-1] == 3:
            img = np.uint8(imresize(img, (viewsize[0], viewsize[1], 3)))
        else:
            img = np.uint8(imresize(img.squeeze(), viewsize[:2]))

        batchimage[map == i] = img.flatten()

    return batchimage



def preprocess_input(im_file, scale=255.):
    """ Read image and (eventually) scale data
        Arguments:
            im_file     : input image file
            scale       : pixel scaling value
        Returns:
            The image
    """
    img = cv2.imread(im_file, 1)
    im_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    im_y = im_ycbcr[:, :, 0]

    return im_y.astype(np.float32) / scale


def window_decision(img_data, model, coeff_map, win_size=(64, 64), stride=8):
    """Divide input image into blocks then classify each block with a trained CNN model. Block scores are aggregated
       by means of average soft decision **on probability of being contrast enhanced**.

    Args:
       img_data: image matrix.
       model: trained CNN model (Keras)
       win_size: (height, width) of the image blocks. Default: (64, 64)

    Returns:
       Score
    """

    # Make sure image has 3-channels
    # if len(img_data.shape) == 2:
    #     img_data = np.tile(np.reshape(img_data, (img_data.shape[0], img_data.shape[1], 1)), (1, 1, 3))

    img_data = np.reshape(img_data, (img_data.shape[0], img_data.shape[1], 1))

    # Divide image into overlapping blocks
    image_view = view_as_windows(img_data, [win_size[0], win_size[1], 1], stride)

    h, w, c, _, _, _ = image_view.shape

    # Reshape the image view so that it is a stack of size N_blocks. Each element is a color patch BxBx3
    slices = image_view.reshape(h * w * c, win_size[0], win_size[1], 1)

    # Test the stack
    predicted_values = model.predict(slices, verbose=0)

    labels = batch_label2_coefficient(predicted_values, coeff_map)

    # Soft decision on each block (label = 1 means manipulated image)
    n_blocks = predicted_values.shape[0]

    # return np.sum(predicted_values[:, 1]) / n_blocks
    return labels


if __name__ == '__main__':

    input_img_dir = 'localisation/images/'

    list_images = glob(os.path.join(input_img_dir, '*.jpg'))

    # Max value for coefficients
    max_coeffs, _ = max_min_coefficient(quality_range=(50, 100),
                                        n_coeffs=cfg.max_no_Q_coefs,
                                        zig_zag_order=True)

    model_file = 'models/model_QF1_60-98-s1-2-term-loss-from-86+20+4+30+ep-30-coef-15/model_ep29.h5'
    model = load_model(model_file,
                           custom_objects=({'custom_softmax': custom_softmax_activation(max_coeffs),
                                            'custom_categorical': custom_categorical(max_coeffs)}))

    # Localisation
    for idx, fimg in enumerate(list_images):

        imgname = os.path.splitext(os.path.basename(fimg))[0]
        map_path = input_img_dir

        if idx>0:
            continue
        #if os.path.exists(map_path):
        #    os.removedirs(map_path)
        #os.makedirs(map_path, True)

        print('Processing localisation for {}. Image {} of {}'.format(fimg, idx+1, len(list_images)))

        # Read and preprocess image (same as training)
        img = preprocess_input(fimg)

        # Perform localisation
        all_maps = window_decision(img, model, max_coeffs, (64, 64, 1), stride=8)

        map_shape = all_maps[:, 0].shape
        map_stack = np.zeros((int(np.sqrt(map_shape[0])), int(np.sqrt(map_shape[0])), all_maps.shape[1]))

        for i in np.arange(0, 14):
            map_ci = all_maps[:, i]
            map_ci = np.reshape(map_ci, (map_ci.shape[0], 1))
            map_ci = np.reshape(map_ci, (int(np.sqrt(map_shape[0])), int(np.sqrt(map_shape[0]))))
            map_stack[:, :, i] = map_ci

            # Save localisation map for each coefficient
            plt.imsave('{}_localisation_map_c{}.png'.format(os.path.join(map_path, os.path.splitext(imgname)[0]), i), map_ci, cmap='jet')

            #fig = plt.figure(figsize = (map_ci.shape[0], map_ci.shape[1])) # Your image (W)idth and (H)eight in inches
            # Stretch image to full figure, removing "grey region"
            #plt.subplots_adjust(left = 0, right = 1, top = 1, bottom = 0)
            #im = plt.imshow(map_ci) # Show the image
            #pos = fig.add_axes([0.93,0.1,0.02,0.35]) # Set colorbar position in fig
            #fig.colorbar(im, cax=pos) # Create the colorbar
            #plt.savefig('{}_localisation_map_c{}.png'.format(os.path.join(map_path, os.path.splitext(imgname)[0]), i))

        # Save stack before final resize to original image size
        np.save('{}_stack.npy'.format(os.path.join(map_path, os.path.splitext(imgname)[0])), map_stack)

        #map_c0 = all_maps[:,0]
        #map_c0 = np.reshape(map_c0, (map_c0.shape[0], 1))
        #map_shape = map_c0.shape
        #map_c0 = np.reshape(map_c0, (int(np.sqrt(map_shape[0])), int(np.sqrt(map_shape[0])), map.shape[1]))

        #map_c0 = map_stack[:,:,0]
        #map_im = np.uint8(resize(map_c0, (img.shape[0], img.shape[1], 1)))

        #plt.imsave('{}_map.png'.format(imgname), map_im[:, :, 0], cmap='jet')

        print('Done!')
