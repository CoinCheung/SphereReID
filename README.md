# SphereReID

This is my implementation of [SphereReID](https://arxiv.org/abs/1807.00537).

My working environment is python3.5.2, and my pytorch version is 0.4.0. If things are not going well on your system, please check you environment.

I only implement the *network-D* in the paper which is claimed to have highest performance of the four networks that the author proposed. 

To train the model, just run the training script:  
```
    $ python train.py
```
This will train the model and save the parameters to the directory of ```res/```.
