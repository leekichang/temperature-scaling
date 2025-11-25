# Temperature-Scaling for Federated Learning

This repository provides a clean PyTorch implementation of **temperature scaling (logit scaling / chilling)** applied to **local training in Federated Learning (FL)**. It includes a lightweight FL simulator with multiple datasets, models, and aggregation methods. The core goal is to study how scaling the logits during local updates improves convergence speed and final accuracy in non-iid FL settings.

---

## 1. Overview

Federated Learning often suffers from slow or unstable convergence under **non-iid client data**.  
This project modifies the **local training objective** by scaling logits before softmax + cross-entropy:

\[
\tilde{z} = \frac{z}{T}
\]

- **T < 1** → amplified logits → more confident gradients → faster convergence  
- **T > 1** → flattened logits → more conservative training  
- Proper selection of T can significantly improve **convergence speed**, **final accuracy**, or **stability** depending on the dataset.

This implementation supports:

- Multiple datasets (MNIST, EMNIST, FashionMNIST, CIFAR10, CIFAR100)
- Multiple models (MLP, CNN, ResNet18)
- FL aggregators (FedAvg, Median, TrimmedMean, Krum)
- Dirichlet-based non-iid splitting
- Easy experiment automation for different temperatures

---

## 2. Installation

### Requirements
- Python ≥ 3.8  
- PyTorch  
- torchvision  
- numpy  
- matplotlib  
- seaborn  
- tqdm  
- tensorboard (optional)

Install:

```
pip install torch torchvision numpy matplotlib seaborn tqdm tensorboard
````

---

## 3. Project Structure

```
temperature-scaling/
├── Clients/            # Client implementations (e.g., NaiveClient)
├── Servers/            # FL server logic and aggregators
├── DataManager/        # Dataset loading + Dirichlet splitting
├── models/             # MLP, CNN, ResNet18 architectures
├── utils.py            # Argument parser, optimizer builder, logging utilities
├── config.py           # Dataset-specific constants
├── example.ipynb       # End-to-end usage example
└── README.md
```

Datasets are automatically downloaded to:

```
../dataset/
```

---

## 4. Quick Start

### A. Using Jupyter Notebook (Recommended)

```bash
jupyter notebook example.ipynb
```

The notebook includes:

* End-to-end FL training
* Accuracy/loss curves
* Temperature comparisons

### B. Running with Python Script

Create a script (e.g., `run_flexchill.py`):

```python
import numpy as np
import torch

import utils
from Servers import BaseServer

if __name__ == "__main__":
    args = utils.parse_args()

    args.dataset = "MNIST"
    args.model = "MLP"
    args.T = 0.5   # temperature scaling factor

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    server = BaseServer(args)
    acc_trace, loss_trace = server.run()

    print("Final accuracy:", acc_trace[-1])
    print("Final loss:", loss_trace[-1])
```

Run:

```bash
python run_flexchill.py --rounds 50 --n_clients 100 --p_ratio 0.1
```

---

## 5. Command-Line Arguments

Below are the main arguments defined in `utils.parse_args()`:

| Argument       | Description                                 | Default       |
| -------------- | ------------------------------------------- | ------------- |
| `--exp_name`   | Experiment name                             | `Template`    |
| `--model`      | `MLP`, `CNN`, `ResNet18`                    | `MLP`         |
| `--dataset`    | Dataset choice                              | `MNIST`       |
| `--optimizer`  | SGD, Adam, etc.                             | `SGD`         |
| `--lr`         | Learning rate                               | `1e-3`        |
| `--decay`      | Weight decay                                | `1e-4`        |
| `--batch_size` | Local mini-batch size                       | `64`          |
| `--epoch`      | Local epochs per round                      | `10`          |
| `--seed`       | Random seed                                 | `0`           |
| `--n_clients`  | Total number of clients                     | `50`          |
| `--rounds`     | Total FL rounds                             | `100`         |
| `--alpha`      | Dirichlet α (non-iid severity)              | `0.5`         |
| `--p_ratio`    | Fraction of participating clients per round | `0.2`         |
| `--client`     | Client class                                | `NaiveClient` |
| `--aggregator` | `FedAvg`, `Median`, `TrimmedMean`, `Krum`   | `FedAvg`      |
| `--T`          | **Temperature scaling factor**              | `1.0`         |

---

## 6. Temperature Scaling

During local training, the model applies:

[
\text{softmax}\left(\frac{z}{T}\right)
]

Effects:

### **T < 1 (stronger gradients)**

* sharper logits
* more confident predictions
* faster convergence
* often better early-round accuracy

### **T > 1 (softer gradients)**

* smoother logits
* more stable training
* potentially better global generalization in highly non-iid setups

Empirically, **T ∈ {0.05, 0.25, 0.5, 1.0}** works well.

---

## 7. Logging & Checkpoints

Global model checkpoints:

```
./checkpoints/<exp_name>/
```

TensorBoard logs (if enabled via `--use_tb True`):

```
./tensorboard/<timestamp>_<exp_name>/
```

Tracked metrics:

* Accuracy per round
* Loss per round
* Time per round
* Temperature comparison curves

---

## 8. Citation

If you use this repository in your research, please cite:

```bibtex
@article{lee2024temperature,
  title   = {Improving Local Training in Federated Learning via Temperature Scaling},
  author  = {Lee, Kichang and Kim, Songkuk and Ko, JeongGil},
  journal = {arXiv preprint arXiv:2401.09986},
  year    = {2024}
}
```

---
## 9. Contact
For questions, issues, or suggestions, feel free to open a GitHub issue or contact the authors.
---
