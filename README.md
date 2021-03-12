## *β*-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints for Neural Network Verification

Robustness verification of neural networks aims to formally prove the correct
prediction staying unchanged for any bounded adversarial perturbations.  Many
existing verifiers solve a relaxed problem such as linear programming (LP) to
give a sound but incomplete solution.  For complete verification, branch and
bound (BaB) with per-neuron split constraints can be used in conjunction with
an incomplete verifier (such as a LP solver).

*β*-CROWN encodes per-neuron split constraints used in branch-and-bound in an
**efficient and highly parallelizable bound propagation** manner (analogous to
[CROWN](https://arxiv.org/pdf/1811.00866.pdf)) **without relying on any
computationally extensive LP solvers**.  Thanks to its efficient bound
propagation procedure, *β*-CROWN can be implemented on GPUs to significantly
speedup branch and bound based formal verification.  Additionally, *β*-CROWN
bounds can be optimized cheaply using gradients obtained by autodiff, allowing
it to obtain **tighter bounds than typical LP verifiers** when intermediate layers' bounds are
jointly optimized.

*β*-CROWN is very fast and also produces better bounds than existing complete and
incomplete neural network verifiers in several settings:

- For complete verification, we are about **three magnitudes faster** than the
  traditional linear programming (LP) based branch and bound approach [(Ehlers,
  2017)](https://arxiv.org/abs/1705.01320), and also outperform
  state-of-the-art GPU-based verifiers such as
  [Fast-and-Complete](https://arxiv.org/pdf/2011.13824.pdf) (ICLR 2021) and
  [Active Sets](https://openreview.net/pdf?id=uQfOy7LrlTR) (ICLR 2021).
- The efficiency of *β*-CROWN allows us to terminate branch and bound early for
  incomplete verification.  Compared to incomplete verifiers with tight
  relaxations such as [SDP-FO](https://arxiv.org/abs/2010.11645) (NeurIPS
  2020), we achieve better verified errors on the very challenging setting of
  **verification agnostic models** (purely adversarially trained). On MNIST
  with eps=0.3, we reduce the gap between PGD error (upper bound) and verified
  error (lower bound) from 35% (SDP-FO) to 15% (*β*-CROWN).


![comparison of bounds for incomplete verification](http://www.huan-zhang.com/images/paper/beta_crown_incomplete.png)

The two figures above show the verification lower bounds (y-axis) computed by
*β*-CROWN BaBSR (**3 minutes** per datapoint),
[SDP-FO](https://github.com/deepmind/jax_verify) (**2 to 3 hours** per
datapoint) and linear programming (LP), as well as the upper bounds (x-axis)
computed by PGD adversarial attacks.  Higher quality bounds are closer to y=x
(black line).  On these adversarially trained models, a typical LP based
verifier gives quite loose bounds. *β*-CROWN BaBSR gives similar or better
quality of bounds compared to SDP-FO while using only a fraction of time.

Details of *β*-CROWN can be found in [our paper](https://arxiv.org/pdf/2103.06624.pdf):

```
@article{wang2021betacrown,
  title={Beta-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints for Complete and Incomplete Neural Network Verification},
  author={Wang, Shiqi and Zhang, Huan and Xu, Kaidi and Lin, Xue and Jana, Suman and Hsieh, Cho-Jui and Kolter, Zico},
  journal={arXiv preprint arXiv:2103.06624},
  year={2021}
}
```


### Installation

Our implementation utilizes the [auto_LiRPA](https://github.com/KaidiXu/auto_LiRPA)
library which is a high quality implementation of
[CROWN](https://github.com/huanzhang12/RecurJac-and-CROWN) and other linear
relaxation based bound propagation algorithms on general neural network
architectures with GPU acceleration.


To run our code, please first clone our repository and install dependencies:


```bash
git clone https://github.com/KaidiXu/Beta-CROWN
git submodule update --init
pip install -r requirements.txt
cd src  # All code files are in the src/ folder
```

(Optinal)
We also provide a `environment.yml` file for creating a conda environment with necessary packages:

```bash
conda env create -f environment.yml
conda activate betacrown
```

Our code is tested on Ubuntu 20.04 with PyTorch 1.7 and Python 3.7.


### Complete verification

We use the set of CIFAR-10 models (base, wide and deep) and accordingly properties provided in [VNN Comp
2020](https://github.com/verivital/vnn-comp/tree/master/2020/CNN/).  This set
of models and properties have become the standard benchmark in a few papers in complete
verification.

To reproduce our results, for example, on CIFAR-10 Base model, please run:

```bash
python bab_verification.py --device cuda --load "models/cifar_base_kw.pth" --model cifar_model --data CIFAR --batch_size 400 --timeout 3600 --mode complete
```

On CIFAR-10 Wide model:

```bash
python bab_verification.py --device cuda --load "models/cifar_wide_kw.pth" --model cifar_model_wide --data CIFAR --batch_size 200 --timeout 3600 --mode complete
```

On CIFAR-10 Deep model:

```bash
python bab_verification.py --device cuda --load "models/cifar_deep_kw.pth" --model cifar_model_deep --data CIFAR --batch_size 150 --timeout 3600 --mode complete
```

After finishing running the command, you should see the reported mean and median of the running time and the number of branches for all properties in the dataset.


### Incomplete verification

For the incomplete verification setting, we obtain two adversarially trained
models (CNN-A-Adv for MNIST and CNN-B-Adv for CIFAR-10) from [Dathathri et al.
2020](https://arxiv.org/abs/2010.11645) which were used to evaluate SDP-FO.
These models were purely PGD adversarially trained (without a certified defense)
and the strong and expensive semidefinite programming based relaxation (which
SDP-FO is based on) was the only possible way to obtain good verified
accuracy before.

To reproduce our results for incomplete verification (verification lower bound and verified accuracy), for example, on MNIST CNN-A-Adv, please run:

```bash
python bab_verification.py --device cuda --load "models/mnist_cnn_a_adv.model" --model mnist_cnn_4layer --data MNIST_SAMPLE --batch_size 300 --timeout 180 --mode incomplete
python bab_verification.py --device cuda --load "models/mnist_cnn_a_adv.model" --model mnist_cnn_4layer --data MNIST_SAMPLE --batch_size 300 --timeout 180 --mode verified-acc
```

On CIFAR-10 CNN-B-Adv:

```bash
python bab_verification.py --device cuda --load "models/cifar_cnn_b_adv.model" --model cnn_4layer_b --data CIFAR_SAMPLE --batch_size 32 --timeout 180 --mode incomplete
python bab_verification.py --device cuda --load "models/cifar_cnn_b_adv.model" --model cnn_4layer_b --data CIFAR_SAMPLE --batch_size 32 --timeout 180 --mode verified-acc
```

Note that `--mode` has three options: "complete", "incomplete" and "verified-acc". Setting `--mode` to "incomplete" to run incomplete verification with the specific timeout and it will keep tightening the verification lower bound until the timeout threshold is reached (it will not stop even if the property is verified, and attempts to find a lower bound as tight as possible). One can also obtain verified accuracy using `--mode verified-acc` which verifies against all labels. Our *β*-CROWN BaBSR can achieve 66% verified accuracy for MNIST CNN-A-Adv and 35% for CIFAR10 CNN-B-Adv on these randomly sub-sampled datasets: we sampled 100 images out of MNIST and CIFAR10 datasets that are correctly classified by MNIST CNN-A-Adv and CIFAR10 CNN-B-Adv models, stored in `src/data/sample100_unnormalized/`. To load and run with these sampled 100 images, one can use `--data MNIST_SAMPLE` or `CIFAR_SAMPLE` as shown in the examples above.


