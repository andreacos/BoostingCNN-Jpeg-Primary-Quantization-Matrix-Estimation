# BoostingCNN-Jpeg-Primary-Quantization-Matrix-Estimation
Boosting CNN-based primary quantization matrix estimation of double JPEG images via a classification-like architecture

![Image](./resources/vippdiism.png)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
more details. You should have received a copy of the GNU General Public License along with this program.
If not, see <http://www.gnu.org/licenses/>.

If you are using this software, please cite:

    Boosting CNN-based primary quantization matrix estimation of double JPEG images via a classification-like architecture}, 
    Benedetta Tondi and Andrea Costanzo and Dequ Huang and Bin Li
    ArXiv preprint: https://arxiv.org/abs/2012.00468
    
The software estimates the primary quantization matrix of a Double JPEG image (either aligned and not aligned) 
based on a Convolutional Neural Network. The CNN-based estimator works with a 64x64 input patch size. 
The first 15 coefficients of the primary quantization matrix, in zig zag order, are returned by the software.

A model is trained for a fixed quality of the second JPEG compression QF2.

<br>

## Installing dependencies

To install the packages required by our software, you may use the provided *requirements.txt*:
```
cd BoostingCNN-Jpeg-Primary-Quantization-Matrix-Estimation
python3 -m venv myvenv
source myvenv/bin/activate
pip install -r resources/requirements.txt
```
We tested our codes on Python 3.5 and Python 3.6 under Ubuntu 16.04 and 18.04 (64 bit).

<br>

## Preparing datasets

We used a copy of the [RAISE8K dataset](http://loki.disi.unitn.it/RAISE/), which was preliminarily 
(and randomly) split into two sub-folders:
* **Train**: 7000 TIF images
* **Test**: 1157 TIF images

The lists of images composing each sub-set can be found in the *./resources* folder: files are named 
respectively *r8k_train.txt* and *r8k_test.txt*. To split your own copy of the RAISE8K dataset, you can use 
the following script:

```
cd CnnJpegPrimaryQuantizationEstimation
python3 -m venv myvenv
source myvenv/bin/activate
python3 Split_RAISE8K.py -rdir "/path/to/full/raise8k" -o "output/path/to/split/raise8k" -res "./resources" -copy 1
```

where: *-rdir* is the path to the RAISE8K directory; *-o* is the path to the directory where RAISE8K is split
into two subfolders (Train, Test); *-res* is the directory where .txt lists are stored; *-copy* either
duplicates (1) or moves (0) files


#### Dataset configuration

The creation of the datasets of image patches is controlled by the parameters listed in *configuration.py* 
(also reported in this document); default values in the file coincide with those we used in our paper's 
implementation.

The script *dataset.py* creates a training and test dataset of double JPEG compressed image patches 
for a given set of JPEG quality factor pairs (QF1, QF2). The first JPEG compression's quality factor 
varies in a given range (e.g. QF1 = [60, 70, 80, 90]) while the second JPEG compression's quality factor is 
fixed (e.g. QF2 = 90). The code also creates two CSV files that contain information about all patches 
for all (QF1, QF2) pairs; such information, specifically the path to the image files and their training labels 
(i.e. the JPEG quantization matrix flattened array in zig-zag order), is used to drive the training and 
test phases.

Some settings in *configuration.py* are self-explaining. The most important settings are the following:

Set the patch size and the number of patches that are (randomly) picked from each input image. The fixed number
of blocks for each image avoids to create a biased dataset from too few images:
```
block_size = (64, 64)          # Image size for the whole training / testing process
max_blocks_img = 100           # Number of blocks that are created for each image
```
Set input folders where the original RAISE 8K dataset has been split with the *Split_RAISE8K.py* utility:
```
input_train_dir = '/Datasets/RAISE8Ksplit/Train'
input_test_dir = '/Datasets/RAISE8Ksplit/Test'
```
Set output folders where the training and test patches and their CSV companions are created, one folder
for each (QF1, QF2) pair, one CSV for joining them all:
```
out_train_dir = '/Datasets/DeepQuantiFinder/Train'
out_test_dir = '/Datasets/DeepQuantiFinder/Test'
```
Set the name of the CSV files used to drive the training and test phases:
```
training_csv = 'train.csv'
test_csv = 'test.csv'
```
Set the (QF1, QF2) pairs and the number of training and test images for each QF pair:
```
# For ALL QF_1 and QF_2 = 90 setup
q_factors = [(60, 90), (61, 90), (62, 90), (63, 90), (64,  90), (65, 90), (66, 90), (67, 90), (68, 90), (69, 90),
             (70, 90), (71, 90), (72, 90), (73, 90), (74,  90), (75, 90), (76, 90), (77, 90), (78, 90), (79, 90),
             (80, 90), (81, 90), (82, 90), (83, 90), (84,  90), (85, 90), (86, 90), (87, 90), (88, 90), (89, 90),
             (90, 90), (91, 90), (92, 90), (93, 90), (94,  90), (95, 90), (96, 90), (97, 90), (98, 90)]

# For ALL QF_1 and QF_2 = 80 setup
q_factors = [(55, 80), (56, 80), (57, 80), (58, 80), (59, 80),
             (60, 80), (61, 80), (62, 80), (63, 80), (64, 80), (65, 80), (66, 80), (67, 80), (68, 80), (69, 80),
             (70, 80), (71, 80), (72, 80), (73, 80), (74, 80), (75, 80), (76, 80), (77, 80), (78, 80), (79, 80),
             (80, 80), (81, 80), (82, 80), (83, 80), (84, 80), (85, 80), (86, 80), (87, 80), (88, 80), (89, 80),
             (90, 80), (91, 80), (92, 80), (93, 80), (94, 80), (95, 80), (96, 80), (97, 80), (98, 80)]
             
n_blocks_train = [1e5]*len(q_factors)
n_blocks_test = [1e4]*len(q_factors)
```
To create the dataset each input image from the RAISE8K dataset is converted to the YCbCr color space. Then, the luminance channel is split into blocks of size (64, 64). The training label corresponding to each block is the zig-zag order array of the JPEG quantization coefficients for the luma channel of the input RGB image. Therefore, network input size is (batch_size, 64, 64, 1).

#### Model training and testing

To train a model with the proposed two-terms loss, run the *train_2terms.py* script. Model training is controlled by the parameters listed 
in *configuration.py*; default values in the file coincide with those we used in our paper's implementation.
```
base_lr = 1e-5                         # Learning rate
max_no_Q_coefs = 15                    # Number of quantization coefficients used for training
batch_size = 32                        # Training batch size
n_epochs = 40                          # Training epochs
scaling_factor_data = 255.0            # Input images (values [0, 255]) are scaled to [0, 1]
snapshot_frequency = 1000              # Frequency (iterations for saving training metrics)
```
In *train_2terms.py* is possible to use other networks. We used DenseNet, whose implementation in this Git is 
the one by Christopher Masch [available on GitHub](https://github.com/cmasch/densenet/blob/master/densenet.py).      

To test the model you can use *predict_2terms.py*. Make sure to adapt the script to point to the right 
to-be-tested model file:  
```
model_file = '/path/to/the/model/the_model.h5'
``` 
The script computes some metrics including average Mean Square Error (MSE), average Normalised MSE and average 
accuracy from CNN's estimated coefficients and ground truth for each pair (QF1, QF2). The script also generates
a CSV file with all the details of the test for each image of the dataset.

Note that *train_1term.py* and *predict_1term.py* are the implementations of the [original technique](https://github.com/andreacos/CnnJpegPrimaryQuantizationEstimation)

#### Usage example
The script [usage.py](https://github.com/andreacos/BoostingCNN-Jpeg-Primary-Quantization-Matrix-Estimation/blob/master/usage.py) contains an example of how to test the provided pre-trained models on a single image. The code is a minimal version of the testing scripts which loads the (QF_1, QF_2) mapping from a Numpy .npy file (provided in the *resources* directoru rather than as a csv.

```    
img_file = 'resources/00000000_redaf7d93t.TIF_85_90.png'
# img_file = 'resources/00000000_redaf7d93t.TIF_50_80.png'
``` 
The image name contains the (QF_1, QF_2) pair used to create it. We provide one image for QF_2=90 and one for QF_2=80.

```  
qf_map = np.load('resources/qf1_qf2_map_90.npy', allow_pickle=True)
# qf_map = np.load('resources/qf1_qf2_map_80.npy', allow_pickle=True)
```  
These are the mapping files for QF_2=90 and QF_2=80.
The predictions layer has shape (1, 226), that is the total length of the one-hot encoded 15 coefficients that are being estimated by the network. The function *label2coefficient()* used to decode each one-hot encoded predictions based on the maximum value that can be assigned to each JPEG coefficient.

```
print(prediction)
print(prediction.shape)
predicted_label = label2coefficient(prediction.flatten(), max_coefficients=max_coeffs)
print(predicted_label)
print(len(predicted_label))

[[2.98665930e-02 5.76886944e-02 6.48024827e-02 1.15710057e-01
  1.55256122e-01 1.28645435e-01 3.38997573e-01 5.85532859e-02
  4.75912951e-02 2.72930181e-03 3.40332117e-05 9.91968022e-07
.......
  2.75804603e-04 5.43303804e-06 6.97390760e-07 8.56585913e-08
  8.70631354e-08 1.60671625e-05 1.40037682e-05 1.23338150e-05
  1.09606153e-05 9.81615358e-06]]
(1, 226)
[ 7  5  4  6  4  3  7  6  4  6  8  7  7  8 11]
15
```

#### Tampering localization
We provide an example of sliding window-based localization in [localization.py](https://github.com/andreacos/BoostingCNN-Jpeg-Primary-Quantization-Matrix-Estimation/blob/master/localization.py). Use this code and the two images in the *resources* folder to reproduce the results in Fig. 12 of the paper. We divide the input image into overlapping blocks of size 64x64 with stride s = 1; then each block is fed to one of the provided pre-trained model that returns a vector with the first Nc estimated decoded quantization coefficients. 

For both tampered images, we have two distinct tampered areas, where the copy-pasted regions have different first JPEG qualities, that is, QF1,1 = 95 and QF1,2 = 85 for the first example and QF1,1 = 65 and QF1,2 = 95 for the second one. The first JPEG quality for the background of the two examples is 75. The last quality factor for both examples is QF2 = 90. All the JPEG grids are not aligned. 

![Image](./resources/T_04_758595.jpg)
![Image](./resources/T_04_758595_localization_map_coeff1.png)
![Image](./resources/T_04_758595_localization_map_coeff6.png)
![Image](./resources/T_04_758595_localization_map_coeff14.png)

![Image](./resources/T_15_859565.jpg)
![Image](./resources/T_15_859565_localization_map_coeff1.png)
![Image](./resources/T_15_859565_localization_map_coeff6.png)
![Image](./resources/T_15_859565_localization_map_coeff14.png)

#### Reproducibility
The choice of image patches when datasets are created is random for each image. Even though the choice of blocks
should not affect the outcome of the training, for sake of reproducibility, we include the list on input images
that we used for training and test in our paper. The lists can be found in ./resources/reproducibility/ and are
called *train_images.txt* and *test_images.txt*.
