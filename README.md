# End-to-end Multi-Camera View Parsing to Top-Down Object Detection on Roadmaps

The goal of this project was to construct a top down view of a road-map and the objects surrounding an ego car using six images taken from a camera mounted on top of a car. We used self-supervised learning to build a representation using a large unlabeled dataset of these 6 multi-view images, and fine-tuned on two downstream tasks: roadmap prediction and bounding box detection. The most obvious use case of this model would be for safe and efficient route planning in autonomous vehicles.

Here is a [3 minute presentation](https://youtu.be/CPdzVIb4RZI) in which I discuss the final approach taken and our model results. You can also read the report uploaded (FinalReport.pdf) for more details about the problem and the various other things tried. This project was done as part of the Deep Learning Spring 2020 class at NYU taught by Professor Yann LeCun and Professor Alfredo Canziani.

# Dataset
The input to the model consists of 6 pictures of a scene taken from 6 cameras mounted on top of the car. The target is a top-down view, or bird's eye view, of the roadmap area surrounding the car, as well as labelled bounding boxes for each of the objects surrounding the car.

Example of an input/output pair: 

![](https://github.com/annikabrundyn/driving-dirty/blob/master/example.png)


# How to run
First, install the dependencies

```python
# clone project   
git clone https://github.com/annikabrundyn/driving-dirty

# install project   
cd driving-dirty

# MAKE SURE TO INSTALL THE APPROPRIATE PYTORCH AND TORCHVISION

pip install -e .   
pip install -r requirements.txt --ignore-installed
```

# Predicting test images
To predict the resulting roadmaps on the test set, run:

```python
cd src/utils
python run_test.py --rm_ckpt_path '../../checkpoints/rm.ckpt'
```


# Training the Autoencoder

```
python src/autoencoder/autoencoder.py --link '/scratch/ab8690/DLSP20Dataset/data' --gpus 1 --max_epochs 5 --batch_size 32
```

Some optional arguments to provide:
```
--link          link to where data is stored
--gpus          how many gpus available
--max_epochs    max number of epochs to train for
```
