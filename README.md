# Deep InfoMax Pytorch

- Python3 + PyTorch 0.41

- Simple Pytorch implementation of Deep InfoMax https://arxiv.org/abs/1808.06670

- Encoding data by maximimizing mutual information between the latent space and in this case, CIFAR 10 images.

- Based on https://github.com/DuaneNielsen/DeepInfomaxPytorch, add my comments for easy understanding

- Official Pytorch implementation [here](https://github.com/rdevon/DIM)


### Part of Results (from https://github.com/DuaneNielsen/DeepInfomaxPytorch)


|              |airplane |automobile | bird | cat |    deer|   dog |    frog|   horse|  ship|   truck|
|-----------------|-------|--------|-------|-------|-------|-------|-------|-------|-------|------|
|Fully supervised |0.7780 | 0.8907 | 0.6233| 0.5606| 0.6891| 0.6420| 0.7967| 0.8206| 0.8619| 0.8291
|DeepInfoMax-Local|0.6120 | 0.6969 | 0.4020| 0.4226| 0.4917| 0.5806| 0.6871| 0.5806| 0.6855| 0.5647
                   

