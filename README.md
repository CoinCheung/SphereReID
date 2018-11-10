# SphereReID

This is my implementation of [SphereReID](https://arxiv.org/abs/1807.00537).

My working environment is python3.5.2, and my pytorch version is 0.4.0. If things are not going well on your system, please check you environment.

I only implement the *network-D* in the paper which is claimed to have highest performance of the four networks that the author proposed. 

### Train and Evaluate
* To train the model, just run the training script:  
```
    $ python train.py
```
This will train the model and save the parameters to the directory of ```res/```.

* To embed the gallery and query set with the trained model and compute the accuracy, directly run:
```
    $ python evaluate.py
```
This will embed the gallery and query set, and then compute cmc and mAP.


### Notes: 
Sadly, I am not able to reproduce the result merely with the method mentioned in the paper.  So I add a few other tricks beyond the paper which help to boost the performance, these tricks includes:   

* During training phase, use [random erasing](https://arxiv.org/abs/1708.04896) augumentation method.

* During embedding phase, aggregate the embeddings of the original pictures and those of their horizontal counterparts by computing the average of these embeddings, as done in [MGN](https://arxiv.org/pdf/1804.01438.pdf).   

* Change the stride of the last stage of resnet50 backbone from 2 to 1.

* Adjust the total training epoch number to 150, and let the learning rate jump by a factor of 0.1 at epoch 90 and 130.

With these tricks, the rank-1 cmc and mAP of my implementation reaches 93.08 and 83.01.
