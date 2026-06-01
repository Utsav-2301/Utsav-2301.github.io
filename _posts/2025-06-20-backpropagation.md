---
layout: post
title: "Backpropagation: The Chain Rule in Disguise"
date: 2025-06-20
category: "Fundamentals"
tags: [backpropagation, chain-rule, gradients, autograd, calculus]
read_time: 12
excerpt: "Backpropagation is the algorithm that makes training deep networks feasible. Strip away the mystique and you'll find it's just the chain rule of calculus, applied systematically to a computation graph."
---

Backpropagation (or "backprop") has a reputation for being hard. It shouldn't. At its heart, it's just one idea: **the chain rule**, applied carefully to a directed computation graph.

Let's build it from the ground up.

## The Chain Rule, Refresher

If $y = f(g(x))$, the chain rule tells us:

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

For a longer chain $y = f_4(f_3(f_2(f_1(x))))$:

$$\frac{dy}{dx} = \frac{\partial f_4}{\partial f_3} \cdot \frac{\partial f_3}{\partial f_2} \cdot \frac{\partial f_2}{\partial f_1} \cdot \frac{\partial f_1}{\partial x}$$

That's it. A neural network is just a very long, structured chain.

## Neural Networks as Computation Graphs

Consider a simple two-layer network. The forward pass computes:

$$\mathbf{z}^{(1)} = \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}$$
$$\mathbf{a}^{(1)} = \text{ReLU}(\mathbf{z}^{(1)})$$
$$\mathbf{z}^{(2)} = \mathbf{W}^{(2)} \mathbf{a}^{(1)} + \mathbf{b}^{(2)}$$
$$\mathcal{L} = \text{CrossEntropy}(\mathbf{z}^{(2)}, y)$$

Backprop works *backwards* through this graph, computing $\frac{\partial \mathcal{L}}{\partial \theta}$ for every parameter $\theta$.

## Step by Step: The Backward Pass

### Step 1: Gradient at the loss

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(2)}} = \hat{\mathbf{p}} - \mathbf{y}$$

(This beautiful result is specific to softmax + cross-entropy: the gradient is just the predicted probabilities minus the one-hot true label.)

### Step 2: Gradient for $\mathbf{W}^{(2)}$ and $\mathbf{b}^{(2)}$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(2)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(2)}} \cdot (\mathbf{a}^{(1)})^T$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(2)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(2)}}$$

### Step 3: Propagate back through layer 1

First, propagate through $\mathbf{W}^{(2)}$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(1)}} = (\mathbf{W}^{(2)})^T \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(2)}}$$

Then through ReLU (its derivative is just a mask — 1 where input > 0, else 0):

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(1)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(1)}} \odot \mathbb{1}[\mathbf{z}^{(1)} > 0]$$

### Step 4: Gradients for $\mathbf{W}^{(1)}$ and $\mathbf{b}^{(1)}$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(1)}} \cdot \mathbf{x}^T$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(1)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(1)}}$$

## A Minimal Implementation from Scratch

```python
import numpy as np

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy(p_hat, y_onehot):
    return -np.mean(np.sum(y_onehot * np.log(p_hat + 1e-9), axis=1))

# Forward pass
def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = np.maximum(0, z1)        # ReLU
    z2 = a1 @ W2 + b2
    p_hat = softmax(z2)
    return z1, a1, z2, p_hat

# Backward pass
def backward(X, y_onehot, z1, a1, z2, p_hat, W2):
    N = X.shape[0]
    
    # Output layer gradient
    dz2 = (p_hat - y_onehot) / N       # (N, 10)
    dW2 = a1.T @ dz2                   # (hidden, 10)
    db2 = dz2.sum(axis=0)              # (10,)
    
    # Hidden layer gradient
    da1 = dz2 @ W2.T                   # (N, hidden)
    dz1 = da1 * (z1 > 0)              # ReLU backward
    dW1 = X.T @ dz1                    # (input, hidden)
    db1 = dz1.sum(axis=0)              # (hidden,)
    
    return dW1, db1, dW2, db2

# One training step
def train_step(X, y_onehot, W1, b1, W2, b2, lr=0.01):
    z1, a1, z2, p_hat = forward(X, W1, b1, W2, b2)
    loss = cross_entropy(p_hat, y_onehot)
    dW1, db1, dW2, db2 = backward(X, y_onehot, z1, a1, z2, p_hat, W2)
    
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    
    return loss, W1, b1, W2, b2
```

## Why Autograd Exists

Manually deriving and coding gradients is tedious and error-prone. Modern frameworks like PyTorch and JAX implement **automatic differentiation** (autograd), which:

1. Records operations during the forward pass into a **computation graph**
2. Traverses the graph backward, applying the chain rule automatically

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 3 + 2 * x            # y = x^3 + 2x
y.backward()
print(x.grad)                  # dy/dx = 3x^2 + 2 = 14.0
```

PyTorch's `requires_grad=True` flags a tensor for tracking. `.backward()` runs backprop. `.grad` holds $\frac{\partial \mathcal{L}}{\partial x}$.

## The Vanishing Gradient Problem

In very deep networks, gradients can shrink exponentially as they flow backward. Each layer multiplies by the derivative of the activation, and if those derivatives are consistently $< 1$ (as with sigmoid), the signal vanishes:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(1)}} = \underbrace{\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(L)}}}_{\text{output grad}} \cdot \prod_{l=2}^{L} \underbrace{\mathbf{W}^{(l)} \cdot \sigma'(\mathbf{z}^{(l-1)})}_{\text{typically} < 1}$$

This is why **ReLU** (derivative = 1 for positive inputs) and **residual connections** (skip connections in ResNets) were game-changers. They provide paths for gradients to flow without shrinking.

## Summary

Backpropagation is the chain rule applied to computation graphs:

1. **Forward pass**: compute activations and loss
2. **Backward pass**: propagate gradients from loss to each parameter
3. **Update**: subtract $\eta \nabla_\theta \mathcal{L}$ from each parameter

It's the foundation of all neural network training. Once you understand it, the "magic" of deep learning starts to feel very mechanical — in the best possible way.

---

> **Next up:** Convolutional Neural Networks — why spatial structure matters, and how convolutions exploit it.
