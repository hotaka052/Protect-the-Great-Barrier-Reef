## TensorFlow - Help Protect the Great Barrier Reef with SSD & DSSD

This repo contains the source code of Kaggle Competion [TensorFlow - Help Protect the Great Barrier Reef](https://www.kaggle.com/competitions/tensorflow-great-barrier-reef) which was held until 2022/2/14.
I have implemented [SSD(Single Shot MultiBox Detector)](https://arxiv.org/abs/1512.02325) & [DSSD(Deconvolutional SSD)](https://arxiv.org/abs/1701.06659) in this scource code.
These models are not state-of-the-art. So, the results of the competition were not good. However, I believe this source code will be good for begginers who want to learn object detection.

## Image

### SSD

![ssd_image](/assets/ssd.png)

### DSSD

![dssd_image](/assets/dssd.png)

### DSSD Feature Map Scale

![dssd_feature](/assets/dssd-feature.png)

## How to train it

I have uploaded the [dataset](https://www.kaggle.com/datasets/suichongmingshi/for-ptgbr) and [Training notes](https://www.kaggle.com/code/suichongmingshi/training-dssd-ptgbr) to kaggle.  
You can train model easily by running this note.

## Reference

https://github.com/amdegroot/ssd.pytorch  
https://arxiv.org/abs/1512.02325  
https://arxiv.org/abs/1701.06659
