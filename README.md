# Certified Data Removal from Machine Learning Models

### Dependencies

torch, torchvision, scikit-learn

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
python train_svhn.py --data-dir <SVHN path> --train-mode private --epsilon 0.1 --delta 1e-5
```

Extracting features using the differentially private extractor:

```bash
python train_svhn.py --data-dir <SVHN path> --saved save/svhn_cnn.pth --test-mode extract --epsilon 0.1
```

### Removing data from trained model

Training a removal-enabled one-vs-all linear classifier and removing 1000 training points:

```bash
python test_removal.py --verbose --extractor dp_eps_0.1 --dataset SVHN --std 10 --lam 2e-4 --num-steps 100 --subsample-ratio 0.1
```

This script randomly samples 1000 training points and applies the Newton update removal mechanism.
The total gradient residual norm bound is accumulated, which governs how many of the 1000 training points can be removed before re-training.
For this setting, the number of certifiably removed training points is limited by the DP feature extractor.

### Reference

This code corresponds to the following paper:

Chuan Guo, Tom Goldstein, Awni Hannun, and Laurens van der Maaten. **[Certified Data Removal from Machine Learning Models](https://arxiv.org/pdf/1911.03030.pdf)**. _arXiv:1911.03030_, 2019.


### Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

### License
This project is CC-BY-NC 4.0 licensed, as found in the LICENSE file.
