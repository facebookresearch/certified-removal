# Certified Data Removal from Machine Learning Models

### Dependencies

torch, torchvision, scikit-learn, pytorch-dp

### Setup

We assume the following project directory structure:

```
<root>/
--> save/
--> result/
```

### Training a differential private (DP) feature extractor

Training a (0.1, 1e-5)-differentially private feature extractor for SVHN:

```bash
python train_svhn.py --data-dir <SVHN path> --train-mode private --std 6 --delta 1e-5 --normalize --save-model
```

Extracting features using the differentially private extractor:

```bash
python train_svhn.py --data-dir <SVHN path> --test-mode extract --std 6 --delta 1e-5
```

### Removing data from trained model

Training a removal-enabled one-vs-all linear classifier and removing 1000 training points:

```bash
python test_removal.py --data-dir <SVHN path> --verbose --extractor dp_delta_1.00e-05_std_6.00 --dataset SVHN --std 10 --lam 2e-4 --num-steps 100 --subsample-ratio 0.1
```

This script randomly samples 1000 training points and applies the Newton update removal mechanism.
The total gradient residual norm bound is accumulated, which governs how many of the 1000 training points can be removed before re-training.
For this setting, the number of certifiably removed training points is limited by the DP feature extractor.

### Removing data from an MNIST 3 vs. 8 model

Training a removal-enabled binary logistic regression classifier for MNIST 3 vs. 8 and removing 1000 training points:

```bash
python test_removal.py --data-dir <MNIST path> --verbose --extractor none --dataset MNIST --train-mode binary --std 10 --lam 1e-3 --num-steps 100
```

### Reference

This code corresponds to the following paper:

Chuan Guo, Tom Goldstein, Awni Hannun, and Laurens van der Maaten. **[Certified Data Removal from Machine Learning Models](https://arxiv.org/pdf/1911.03030.pdf)**. ICML 2020.


### Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

### License
This project is CC-BY-NC 4.0 licensed, as found in the LICENSE file.
