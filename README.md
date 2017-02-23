# faststyle

NOTE: This readme is incomplete.

This repository is a Tensorflow implementation of JC Johnson et al.'s fast style transfer [paper](https://arxiv.org/abs/1603.08155), combined with D. Ulyanov's instance normalization
[paper](https://arxiv.org/abs/1607.08022). It also provides resize-convolution on top of deconvolution for better upsampling as discussed [here](http://distill.pub/2016/deconv-checkerboard/).

The pretrained models at ```faststyle/models``` were used to generate the results below:


## Dependencies

- Python 2.7
- Tensorflow 1.0.0 (with GPU support 
- Numpy
- OpenCV 3.1.0
- (Recommended)

## Setup

If you just intend to utilize the pretrained models, then all you need to do is:
```
git clone https://github.com/ghwatson/faststyle.git
```
If you also intend on training new models, you will need the MS-Coco 13GB training dataset found [here](http://mscoco.org/dataset/#download) and the VGG weights by running:

```
cd faststyle/libs
./get_vgg16_weights.sh
```

To prepare the MS-Coco dataset for use with train.py, you will have to convert it to Tensorflow's TFRecords format, which shards the images into large files for more efficient reading from disk. ```faststyle/tfrecords_writer.py``` can be used for this as shown below. Change ```--num_threads``` to compensate for however many cores you have available, and also such that it divides ```--train_shards```:

```
python tfrecords_writer.py --train_directory /path/to/training/data \
                           --output_directory /path/to/desired/tfrecords/location \
                           --train_shards 126 \
                           --num_threads 6
```

## Usage

The trained models can be used to stylize an image, or your webcam feed.

### ```stylize_image.py```

### ```stylize_webcam.py```

### ```train.py```

### ```slow_style.py```

## Acknowledgements

For the most part, I implemented this repo by using the aforementioned references, as well as Tensorflow's documentation (this was a learning exercise). Furthermore:
- Justin Johnson's [repo](https://github.com/jcjohnson/fast-neural-style) for its documentation, and example images.
- [hzy46/fast-neural-style-tensorflow](https://github.com/hzy46/fast-neural-style-tensorflow) to squash a few bugs.
