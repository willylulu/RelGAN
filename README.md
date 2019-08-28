#   RelGAN (ICCV 2019)

**Keras** implementation of [**RelGAN: Multi-Domain Image-to-Image Translation via Relative Attributes**](https://arxiv.org/abs/1908.07269)

The paper is accepted to ICCV 2019. We also have the PyTorch version [here](https://github.com/elvisyjlin/RelGAN-PyTorch).

##  Preparation
*   Prerequisites
    *   Python  3.5
    *   Keras   2.2.4
*   Dataset
    *   Celeba-HQ
        *   Please follow the instructions in [celeba-hq-modified](https://github.com/willylulu/celeba-hq-modified) to prepare the dataset
*   Pre-trained model
    *   `generator519.h5`
    
##  Get Started

### Preprocessing

In this step, we export annotations to a numpy file. You will get `anno_dic.npy` and `imgIndex.npy` after running the script

```
-n  :   number of attributes (5, 9, 17)
-o  :   target output file
```

```console
python3 preprocessing.py [--number=17] [--output=anno_dic.npy]
```

### Training

```console
python3 train.py
    --path=<path to celeba-256>
    --device=<device number>
    [--growth=False]
    [--step=0]
    [--lr=1e-5]
    [--beta1=0.5]
    [--beta2=0.999]
    [--batch_size=4]
    [--sample_size=2]
    [--epochs=400000]
    [--lambda1=10]
    [--lambda2=10]
    [--lambda4=10]
    [--lambda5=10]
    [--lambda_gp=150]
    [--img_size=256]
    [--vec_size=17]     #if you change the number of attributes, change this number
```

### Testing

```console
python3 demo_translation.py --device=<device number>
python3 demo_interpolation.py --device=<device number>
```