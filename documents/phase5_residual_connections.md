# Phase 5 — Residual Connections (Skip Connections)

## What you already know

```
SimpleCNN (3 CNNBlocks, no skip):    ~64% baseline → ~80% after 40 epochs
TunedCNN (5 CNNBlocks, no skip):     ~87% validation
ResNet18 (18 ResBlocks, skip):       ~85%+ transfer learning

Same dataset, same task.
ResNet18 achieves more BECAUSE of skip connections.
Phase 5 is understanding WHY.
```

---

## CNNBlock vs ResidualBlock

### CNNBlock — plain block

```
x → Conv → BatchNorm → ReLU → MaxPool → out

The input x is transformed and thrown away.
Output depends ONLY on the convolution result.
```

```python
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.block(x)    # input x is gone, only conv result remains
```

### ResidualBlock — block with skip connection

```
x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU → MaxPool → out
                                    ↑
                                 skip (identity)

The input x is PRESERVED and added back.
Output = convolution result + original input.
```

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        residual = self.shortcut(x)              # save input (with channel projection)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual                           # ← SKIP CONNECTION: add input back
        out = F.relu(out)
        out = self.pool(out)                      # downsample after skip
        return out
```

### The key difference in one line

```
CNNBlock:      out = F(x)          → must learn ENTIRE transformation
ResidualBlock: out = F(x) + x      → only needs to learn DELTA from input
```

---

## Why residual is easier to learn

### The identity insight

Imagine a layer where the best thing to do is nothing (pass input through unchanged):

```
CNNBlock must learn:     F(x) = x     → HARD
  Convolutions aren't natural identity functions
  The network has to carefully tune weights to output what was input

ResidualBlock just needs: F(x) = 0     → EASY
  Push all weights to zero
  out = 0 + x = x   ← identity achieved automatically
```

### Each layer automatically decides if it's useful

```
If a layer is USEFUL:   F(x) learns a real transformation, adds to x
                       out = useful_transform + x

If a layer is USELESS:  F(x) goes to zero (weights → 0)
                       out = 0 + x = x
                       The layer becomes transparent — input passes through

The network PRUNES itself. Useless layers don't hurt.
```

---

## The problem skip connections solve: vanishing gradients

### What happens as networks get deeper

```
Forward pass:  input → L1 → L2 → L3 → L4 → L5 → output → loss

Backward pass (gradient flow):
  Loss says: "output is wrong, adjust weights!"

  Each layer multiplies the gradient by its weight values.
  If weights are small (typical): gradient SHRINKS each layer.

  L5 gets gradient:    0.1     (strong signal — "adjust here!")
  L4 gets gradient:    0.01    (weaker)
  L3 gets gradient:    0.001   (fading)
  L2 gets gradient:    0.0001  (almost gone)
  L1 gets gradient:    0.00001 (basically nothing)

  L1 doesn't learn → L1 outputs garbage → L2 gets garbage → whole network suffers
```

### How skip connections fix this

```
Without skip — gradient has ONE path through all layers:
  Loss → L5 → L4 → L3 → L2 → L1
  Gradient: 0.1 × 0.1 × 0.1 × 0.1 × 0.1 = 0.00001

With skip — gradient has a DIRECT PATH to every layer:
  Loss → L5 → L4 → L3 → L2 → L1
                    ↓     ↓     ↓     ↓     ↓
                  skip  skip  skip  skip  skip
                    ↓     ↓     ↓     ↓     ↓
  Each layer gets: conv_gradient + skip_gradient (direct from loss)

  L5 gets gradient:    0.1
  L4 gets gradient:    0.1  (conv path) + 0.1 (skip path) = 0.2
  L3 gets gradient:    0.1  (conv path) + 0.1 (skip path) = 0.2
  L2 gets gradient:    0.1  (conv path) + 0.1 (skip path) = 0.2
  L1 gets gradient:    0.1  (conv path) + 0.1 (skip path) = 0.2

  Every layer gets a STRONG gradient. No fading.
```

### The historical proof

```
This was a famous result from 2015 (He et al., "Deep Residual Learning"):

Network depth    Without skip     With skip
20 layers        70% accuracy     72% accuracy
56 layers        68% accuracy ←   75% accuracy ← WORSE without skip, BETTER with skip!

The 56-layer network WITHOUT skip got WORSE than 20 layers.
Not because of overfitting — it had higher TRAINING error too.
It simply couldn't learn. Gradients vanished.

With skip connections: 56 layers > 20 layers. Deeper IS better.
This enabled ResNet-50, ResNet-101, ResNet-152 — all trainable.
```

---

## Why MaxPool goes AFTER the skip connection

### The bug you had (and why it crashes)

```python
# WRONG — MaxPool inside the block, before skip
def forward(self, x):
    residual = self.shortcut(x)     # shape: (N, 64, 224, 224) — no MaxPool
    x = self.block(x)               # shape: (N, 64, 112, 112) — has MaxPool!
    x += residual                   # 112×112 += 224×224 → CRASH!

# RIGHT — MaxPool after the skip connection
def forward(self, x):
    residual = self.shortcut(x)     # shape: (N, 64, 224, 224)
    out = self.conv1(x)             # shape: (N, 64, 224, 224) — same size (padding=1)
    out = self.conv2(out)           # shape: (N, 64, 224, 224) — same size (padding=1)
    out += residual                 # 224×224 += 224×224 → works! ✅
    out = self.pool(out)            # shape: (N, 64, 112, 112) — downsample AFTER
```

Rule: **skip connection operates on same spatial dimensions. Downsample after.**

---

## Why two convolutions per block?

```
One conv (CNNBlock):                Two convs (ResidualBlock):
  x → Conv → BN → ReLU → pool        x → Conv → BN → ReLU → Conv → BN
                                      then (+x) → ReLU → pool

  Learns one transformation           Learns two transformations before adding input back

One conv: F(x) has limited capacity   Two convs: F(x) has more capacity
  → the "residual" is shallow            → the "residual" captures richer patterns

Standard ResNet uses 2 convs per block. This is the proven design.
```

---

## Tradeoffs

```
ResidualBlock:                    CNNBlock:
  ✅ Better gradient flow           ✅ Simpler, fewer params
  ✅ Can go deeper (10-18+ layers)  ✅ Faster per forward pass
  ✅ Useless layers become identity  ✅ Less memory
  ✅ Better accuracy at same depth   
  ❌ 2× more params per block       ❌ Accuracy plateaus at ~5-7 layers
  ❌ Slower per forward pass        ❌ Can't go beyond 7-8 layers
  ❌ More VRAM needed               
```

---

## When to use which

```
Use CNNBlock when:
  - Shallow network (2-4 layers)
  - Small dataset (< 10K images)
  - Need fast inference speed
  - Limited VRAM

Use ResidualBlock when:
  - Deep network (5+ layers)
  - Medium/large dataset (20K+ images)
  - Want maximum accuracy
  - Can afford more compute
```

---

## Expected accuracy for your project

```
Your results so far:
  SimpleCNN (3 CNNBlocks):           ~80% test
  TunedCNN (5 CNNBlocks):            ~87% val
  ResNet18 Transfer (18 ResBlocks):  ~85%+ (but overfits: val 87% → test 79%)

Predictions for ResidualTunedCNN (5 ResidualBlocks):

  Best case:    88-90%  — skip connections help gradients, slightly better learning
  Same case:    ~87%    — 5 layers isn't deep enough for skip to matter much
  Worst case:   ~85%    — residual adds params that overfit on 28K dataset

Why not a huge jump?
  Skip connections shine when going DEEP (10-18+ layers)
  At only 5 layers, gradients don't vanish much yet
  The real benefit: you can now try 8, 10, 12 layers

  5 layers with skip:    ~87-90%
  8 layers with skip:    ~89-91%    ← skip connections start paying off
  12 layers with skip:   ~90-93%    ← competing with ResNet18 from scratch
  5 layers without skip: ~87%       ← this is the CEILING, can't go deeper
```

---

## The real value of Phase 5

```
NOT: "skip connections make 5 layers better"
BUT: "skip connections UNLOCK going deeper"

         CNNBlock ceiling         ResidualBlock ceiling
              ~87%                       ~93%+
               ↑                         ↑
          5 layers max              12-18 layers possible
          can't go deeper           can keep stacking

Phase 5 teaches you the building block.
The accuracy gain comes from using it to go deeper.
```

---

## Your file structure after Phase 5

```
src/
├── cnn.py                    ← SimpleCNN (3 CNNBlocks, baseline)
├── cnn_tuned.py              ← TunedCNN (5 CNNBlocks, Optuna-optimized)
├── residual_cnn_tuned.py     ← ResidualTunedCNN (5 ResidualBlocks)
├── train_tuned.py            ← trains TunedCNN
├── train_residual.py         ← trains ResidualTunedCNN
└── ...
```

---

## Key terms

| Term | Meaning |
|------|---------|
| Skip connection | Adding input back to output: `out = F(x) + x` |
| Residual | The difference `F(x) = out - x` — what the layer learned BEYOND identity |
| Identity mapping | When `F(x) = 0`, layer passes input through unchanged |
| Shortcut/projection | 1×1 conv to match channels when `in_channels ≠ out_channels` |
| Vanishing gradient | Gradients shrink to near-zero as they flow through many layers |
| Deep network | Generally 10+ layers — where skip connections become essential |
