# CONDITIONAL DC_GANS

## Introduction:
In this project, I aimed to train a Conditional DC GANs model with Wasserstein-GP method using DC_GANs architecutre with Tensorboard to generate the images of Cifar10 dataset images from random noise.

## Tensorboard:
TensorBoard, along with saving training or prediction images, allows you to save them in TensorBoard and examine the changes graphically during the training phase by recording scalar values such as loss and accuracy. It's a very useful and practical tool.

## Dataset:
- I used the Mnist dataset for this project, which consists of 10 labels (handwritten digits) with total 60k images on train and 10k images on test.

## Models:
- I won't go into the details of Dc Gans and Wasserstain method as I explained them in my previous repository. 
- previous repository: https://github.com/aktasumitt/DC_GANs_with_Wasserstein-GP 
- Conditional Gans: https://arxiv.org/abs/1411.1784

- In this project, we will actually add condition (label) to this model, so that the generated image will be compatible with the label. In other words, we will tell the model what it should generate.

## Train:
- When providing the image to be generated to the model, the label is also provided alongside it.
- This label is passed through an embedding layer, and then the dimensions are adjusted to match the given image before being concatenated and fed into the model.
- This process is done in both the generator and the discriminator. Therefore, the number of in_channels for the first input layers in the generator and discriminator is increased by the embedding_size and 1, respectively.
- The shapes of the embedded labels should be (batch_size, embedding_size, 1, 1) for the generator and (batch_size, 1, img_size, img_size) for the discriminator to enable concatenation on dim=1. 
- Embedding_size of decoder is img_size*img_size , Embedding_Size of generator is custom. (we used 100)
- We will use Adam optimizer with learning_rate 1e-4.
- Training is the same as previous.

## Results:
- We can observe that the generated images converge towards the real data after five epochs.
- There are generated images and graph of values on tensorboard.

## Usage: 
- You can train the model by setting "TRAIN" to "True" in config file and your checkpoint will save in "config.CALLBACKS_PATH"
- Tensorboard files will created into "Tensorboard" folder during training time.
- Then you can generate the images from random noise by setting the "LOAD_CHECKPOINTS" and "TEST" values to "True" in the config file.

