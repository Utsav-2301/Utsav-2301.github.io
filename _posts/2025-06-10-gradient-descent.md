---
layout: post
title: "Gradient Descent: Finding the Bottom of a Bowl"
date: 2025-06-10
category: "Optimization"
tags: [gradient-descent, optimization, learning-rate, SGD, Adam]
read_time: 10
excerpt: "Gradient descent is the engine of all deep learning. Here's how it works, why it works, and the variants (SGD, Mini-batch, Adam) that make it practical at scale."
---

Imagine you're blindfolded on a hilly landscape and want to reach the lowest valley. You can't see, but you can feel which direction the ground slopes downward beneath your feet. So you take a small step in the downhill direction, feel the slope again, and repeat.

That's gradient descent — and it's the engine behind all of deep learning.

## The Loss Surface

Training a neural network means minimizing a **loss function** $\mathcal{L}(\theta)$, where $\theta$ represents all the weights and biases. The loss measures how wrong the network is on the training data.

For regression, a common choice is **Mean Squared Error**:

$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \left( f_\theta(x_i) - y_i \right)^2$$

For classification, we typically use **cross-entropy loss**:

$$\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik} \log \hat{p}_{ik}$$

The loss function defines a surface in parameter space. We want to find the parameters $\theta^*$ that correspond to the lowest point.

## The Gradient

The **gradient** $\nabla_\theta \mathcal{L}$ is a vector that points in the direction of steepest *increase* of the loss. So we move in the *opposite* direction:

$$\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}(\theta)$$

Here $\eta$ (eta) is the **learning rate** — how big a step to take. This is one of the most important hyperparameters in deep learning.

- **Too large**: the optimization diverges, bouncing wildly
- **Too small**: training is painfully slow, may get stuck

## Variants of Gradient Descent

### Batch Gradient Descent
Computes the gradient over the *entire* dataset before each update. Precise, but prohibitively slow for large datasets.

### Stochastic Gradient Descent (SGD)
Updates on *one sample at a time*:
$$\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}(x_i, y_i)$$

Very fast, but very noisy. The "random walk" noise can actually help escape bad local minima!

### Mini-batch SGD
The practical sweet spot — update on a **batch** of $B$ samples (typically 32–256):
$$\theta \leftarrow \theta - \eta \cdot \frac{1}{B} \sum_{i \in \text{batch}} \nabla_\theta \mathcal{L}(x_i, y_i)$$

This is what people mean when they say "SGD" in practice. It balances noise and efficiency, and maps well to GPU parallelism.

## Momentum

Plain SGD can oscillate in narrow valleys. **Momentum** accumulates a velocity vector that helps smooth the trajectory:

$$v \leftarrow \mu v - \eta \nabla_\theta \mathcal{L}$$
$$\theta \leftarrow \theta + v$$

Where $\mu \approx 0.9$ is the momentum coefficient. Think of it as a heavy ball rolling downhill — it resists sharp direction changes.

## Adam: The Modern Default

**Adam** (Adaptive Moment Estimation) is the optimizer used in most modern deep learning. It maintains per-parameter learning rates, adapting them based on the history of gradients:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(1st moment, mean)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(2nd moment, variance)}$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{(bias correction)}$$
$$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Default values: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

Parameters with consistently large gradients get smaller updates; parameters with small gradients get larger updates. This makes Adam robust across diverse architectures.

## Implementation in PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Adam optimizer — sensible defaults for most cases
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()          # Clear old gradients
        output = model(X_batch)        # Forward pass
        loss = criterion(output, y_batch)
        loss.backward()                # Backprop: compute gradients
        optimizer.step()               # Update weights
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

## The Learning Rate is Everything

A quick experiment illustrates the stakes:

| Learning Rate | Behavior |
|---|---|
| `1e-1` | Often diverges |
| `1e-3` | Adam's sweet spot for many tasks |
| `1e-4` | Conservative, stable but slow |
| `1e-6` | Barely moves |

In practice, use a **learning rate scheduler** — start higher and decay over training. `CosineAnnealingLR` or `OneCycleLR` are popular modern choices.

## The Bigger Picture

Gradient descent works not because the loss surface is simple — it's high-dimensional and full of saddle points and flat regions — but because:

1. **Overparameterization** means there are many equivalent minima, and most of them generalize similarly
2. **SGD noise** provides implicit regularization
3. **Depth** creates favorable gradient landscapes compared to shallow networks

The field still doesn't fully understand *why* gradient descent finds solutions that generalize so well. It's one of the beautiful open mysteries of deep learning.

---

> **Next up:** Backpropagation — how we actually compute those gradients through a network.
