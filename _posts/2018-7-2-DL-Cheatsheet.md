---
layout: post
title: DL Cheatsheet
---

## 1. Useful Math Functions

[Read more](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#)

### 1.1 Non-linear Activation Functions
Activation functions are used to add non-linearity to neural networks, or otherwise neuralnets are just giant linear classifiers. Activation functions are **differentiable** because we want to do back-propagation.
![_config.yml]({{ site.baseurl }}/images/dl-cheatsheet/activation_functions.png)

#### 1.1.1 Sigmoid (Logistic)
A sigmoid function is an 'S'-shaped curve. Output is between 0 and 1.

$$ S(x) = \frac{1}{1+e^{-x}} $$

![_config.yml]({{ site.baseurl }}/images/dl-cheatsheet/sigmoid.png)
Sigmoid function can cause a neural network to get stuck at the training time.
The gradients of the sigmoid function is calculated as:
![_config.yml]({{ site.baseurl }}/images/dl-cheatsheet/sigmoid_gradients.png)

#### 1.1.2 tanh (hyperbolic tangent)
Similar to sigmoid, but output is in (-1, 1).

$$tanh(x) = \frac{1 - e^{-2x}}{1+e^{-2x}}$$

#### 1.1.3 ReLUs
Most commonly used activation function.

$$ReLU(x) = max(0, x)$$

A variation of ReLU is leaky ReLU, making the negative inputs not die out that fast.
![_config.yml]({{ site.baseurl }}/images/dl-cheatsheet/relu.jpg)

#### 1.1.4 Softmax
Softmax function maps an array of scores into an array of possibilities:

$$softmax(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

It is called softmax due to that:
(1) give large scores large possibilities -> max
(2) still reserve some possibility for low scores -> soft 
Softmax is usually put at the last layer to indicate possibilities of each class, and the class with highest possibility is chosen as output prediction. Softmax is often combined with cross-entropy loss and the gradients calculation will be discussed later.

### 1.2 Loss Functions

#### 1.2.1 Cross-entropy Loss
Cross-entropy loss, or log loss, measures the performance of a classification model whose output is a probability value b/w 0 and 1. Cross-entropy loss increases as the predicted probability diverges from the actual label. So predicting a probability of 0.011 when the actual observation label is 1 would be bad and result in a high loss value. A perfect model would have a log loss of 0 (when ground truth class has a probability of 1).

![_config.yml]({{ site.baseurl }}/images/dl-cheatsheet/crossentropy.png)

The graph above shows the range of possible loss values given a true observation. As the predicted probability approaches 1, loss slowly decreses. As the predicted probability decreses, however, the loss grows rapidly.

$$CELoss(p) = - \sum_{c}y_clog(p_c)$$

where $$y_c = 1$$ when c is the ground-truth class and 0 otherwise. Thus, it only cares the probability of ground-truth class and does not care about the distribution.

The gradients of a softmax cross-entropy loss:

$$p = softmax(x)$$

$$CE(y, p) = -\sum_i y_i log(p_i) = -log(p_{gt}) = log\sum e^{x_j}- x_{gt}$$

$$\frac{\partial CE}{x_k} =  \frac{e^{x_k}}{\sum_j e^{x_j}} = p_k, k\neq gt$$

$$\frac{\partial CE}{x_{gt}} =  \frac{e^{x_{gt}}}{\sum_j e^{x_j}} -1 = p_{gt} -1$$

$$\therefore \frac{\partial CE}{x} = p - y$$

#### 1.2.2 MSE Loss
Mean squared error (MSE) measures the average of the squares of the errors:

$$MSE(p) = \frac{1}{n}\sum_{i}^{n}(p_i - y_i)^2$$

#### 1.2.3 Hinge Loss
In machine learning , the hinge loss is a loss function used for training classifiers. The hinge loss is used for 'maximum-margin' classification, most notably for support vector machines (SVMs). For an intended output $$y = \pm 1$$ and a classifier score $$p$$, the hinge loss of the prediction $$p$$ is defined as:

$$Hinge(y, p) = max(0, 1-y\cdot p)$$ 

#### 1.2.4 Total Variation Loss

#### 1.2.5 Gram Matrix Loss (Style Loss)

### 1.3 benchmark
#### 1.3.1 PSNR
#### 1.3.2 BLEU

## 2. Optimizers
![_config.yml]({{ site.baseurl }}/images/dl-cheatsheet/Optimizers.gif)
### 2.1 Stochastic gradient desencet
Batch gradient descent will calculate the gradient of the whole dataset but will perform only one update, hence it can be very slow and hard to control for datasets which are extremely large and dont't fit in the memory. 
SGD on the other hand performs a parameter update for each training example, it performs one update at a time:

$$\theta = \theta - \alpha \nabla J(x_i, y_i, \theta)$$

pros:
Due to frequent updates, parameters updates have high variance and causes the loss function to fluctuate to different intensities. This is actually a good thing because it helps us discover new and possible better local minima.

cons:
Frequent updates and fluctuations make it complicates the convergence to the exact minimum and will keep overshooting, although if we decrease learning rate gradually it will converge.

![_config.yml]({{ site.baseurl }}/images/dl-cheatsheet/SGD.png)

To improve, we can use mini batch gradient descent.

### 2.2 Momentum
The high variance oscillations in SGD makes it hard to reach convergence, so a technique called Momentum was invented which accelerates SGD by navigating along the relevant direction and softens the oscillations in irrelevant directions. In other words, all it does is adding a fraction $$\gamma$$ of the update vector of the past step to the current update vector:

$$V(t) = \gamma V(t-1) - \alpha\cdot d\theta$$

$$\theta += V(t)$$

where $$\gamma$$ is usually set to 0.9.

### 2.3 Nesterov Accelerated Gradient
NAG improves momentum by slowing the gradient down before it reaches the minimum, or otherwise with large momentum, it will overshoot.
![_config.yml]({{ site.baseurl }}/images/dl-cheatsheet/NAG.jpg)
Nesterov momentum. Instead of evaluating gradient at the current position (red circle), we know that our momentum is about to carry us to the tip of the green arrow. With Nesterov momentum we therefore instead evaluate the gradient at this "looked-ahead" position:

```python
x_ahead = x + mu * v
# evaluate dx_ahead (the gradient at x_ahead instead of at x)
v = mu * v - learning_rate * dx_ahead
x += v
```

### 2.4 Adagrad
Adagrad simply allows the learning rate $$\alpha$$ to adapt based on the parameters. So it makes big updates for infrequent parameters and small updates for frequent parameters. Thus, it is perfect to deal with sparse data.
It uses a different learning Rate for every parameter θ at a time step based on the past gradients which were computed for that parameter.

```python
# Assume the gradient dx and parameter vector x
cache += dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)
```

Notice that the variable cache has size equal to the size of the gradient, and keeps track of per-parameter sum of squared gradients. This is then used to normalize the parameter update step, element-wise. Notice that the weights that receive high gradients will have their effective learning rate reduced, while weights that receive small or infrequent updates will have their effective learning rate increased. Amusingly, the square root operation turns out to be very important and without it the algorithm performs much worse. The smoothing term eps (usually set somewhere in range from 1e-4 to 1e-8) avoids division by zero. A downside of Adagrad is that in case of Deep Learning, the monotonic learning rate usually proves too aggressive and stops learning too early.

### 2.5 RMSprop
RMSprop is a very effective, but currently unpublished adaptive learning rate method. Amusingly, everyone who uses this method in their work currently cites slide 29 of Lecture 6 of Geoff Hinton’s Coursera class. The RMSProp update adjusts the Adagrad method in a very simple way in an attempt to reduce its aggressive, monotonically decreasing learning rate. In particular, it uses a moving average of squared gradients instead, giving:

```python
cache = decay_rate * cache + (1 - decay_rate) * dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)
```

Here, decay_rate is a hyperparameter and typical values are [0.9, 0.99, 0.999]. Notice that the x+= update is identical to Adagrad, but the cache variable is a “leaky”. Hence, RMSProp still modulates the learning rate of each weight based on the magnitudes of its gradients, which has a beneficial equalizing effect, but unlike Adagrad the updates do not get monotonically smaller.

### 2.5 Adam
Adam is a recently proposed update that looks a bit like RMSProp with momentum. The (simplified) update looks as follows:

```python
m = beta1*m + (1-beta1)*dx
v = beta2*v + (1-beta2)*(dx**2)
x += - learning_rate * m / (np.sqrt(v) + eps)
```

Notice that the update looks exactly as RMSProp update, except the “smooth” version of the gradient m is used instead of the raw (and perhaps noisy) gradient vector dx. Recommended values in the paper are eps = 1e-8, beta1 = 0.9, beta2 = 0.999. In practice Adam is currently recommended as the default algorithm to use, and often works slightly better than RMSProp. However, it is often also worth trying SGD+Nesterov Momentum as an alternative. The full Adam update also includes a bias correction mechanism, which compensates for the fact that in the first few time steps the vectors m,v are both initialized and therefore biased at zero, before they fully “warm up”. With the bias correction mechanism, the update looks as follows:

```python
# t is your iteration counter going from 1 to infinity
m = beta1*m + (1-beta1)*dx
mt = m / (1-beta1**t)
v = beta2*v + (1-beta2)*(dx**2)
vt = v / (1-beta2**t)
x += - learning_rate * mt / (np.sqrt(vt) + eps)
```

## 2. techniques

### 2.1 attention mechanism
#### 2.1.1 soft attention
#### 2.1.2 hard attention
### 2.2 batch normalization
### 2.3 dropout
### 2.4 beam search

## 3. models
### 3.1 basic structures
#### 3.1.1 CNN
#### 3.1.2 RNN
#### 3.1.3 LSTM
#### 3.1.4 GRU
#### 3.1.5 ResNet

### 3.2 object detection
#### 3.2.1 R-CNN
#### 3.2.2 Fast R-CNN
#### 3.2.3 Faster R-CNN
#### 3.2.4 Mask R-CNN
#### 3.2.5 YOLO
#### 3.2.6 YOLO2
#### 3.2.7 SSD

### 3.3 others
#### 3.3.1 FCN

