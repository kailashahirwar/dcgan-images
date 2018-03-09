# dcgan-images
Implementation of DCGAN in Keras to generate new images (https://arxiv.org/pdf/1511.06434.pdf)

## Usage

Training mode:

    python main.py --mode train --data_path <directory_path> --batch_size <desired_batch_size>
    
Generation mode:
    
    python main.py --mode generate --batch_size <number_of_images_to_generate>
    
## Command line arguments
    
    --mode             choose between the two modes: (train, generate)
    --batch_size       The size of each batch (default: 128)
    --learning_rate    The learning rate for the Adam optimizers (default: 0.0002)
    --beta_1           The beta 1 value for the Adam optimizers (default: 0.5)
    --epochs           The amount of epochs the network should train (default: 100)
    --data_path        The path to the images that should be used for training
    
## References
    
[DCGAN-nature](https://github.com/Skuldur/DCGAN-Nature)

[DCGAN Research Paper](https://arxiv.org/pdf/1511.06434.pdf)