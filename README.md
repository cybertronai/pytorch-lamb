Implementation of https://arxiv.org/abs/1904.00962 for large batch, large learning rate training.

Bonus: TensorboardX logging (example below).

## Try the sample
```
git clone git@github.com:cybertronai/pytorch-lamb.git
cd pytorch-lamb
pip install -e .
python test_lamb.py
tensorboard 
```

## Sample results
At `--lr=.1`, the Adam optimizer is unable to train. With a little weight decay, LAMB avoids diverging!

Green: `python test_lamb.py --batch-size=512 --lr=.1 --wd=0 --log-interval 30 --optimizer lamb`

Blue: `python test_lamb.py --batch-size=512 --lr=.1 --wd=.01 --log-interval 30 --optimizer lamb`
![](images/loss.png)

`r1` is the L2 norm of the weights. You can see in the green plot that some of the weights start to run away, which leads to divergence. This is why weight decay helps.
![](images/histogram.png)