# *β*-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints for Neural Network Verification

## News

- *β*-CROWN empowered our [*α*,*β*-CROWN verifier](https://github.com/huanzhang12/alpha-beta-CROWN) (alpha-beta-CROWN), which **won VNN-COMP 2021** with the highest total score. Our verifier solves the most number of instances on 8 benchmarks and outperforms 11 SOTA tools. (details of competition results can be found
[in the slides here](https://docs.google.com/presentation/d/1oM3NqqU03EUqgQVc3bGK2ENgHa57u-W6Q63Vflkv000/edit#slide=id.ge4496ad360_14_21) and [the report here](https://arxiv.org/abs/2109.00498)).
- We have released a new codebase and you should use the  [*α*,*β*-CROWN repository](https://github.com/huanzhang12/alpha-beta-CROWN) to reproduce the experiments on our paper. See detailed instructions below.

## Introduction

Robustness verification of neural networks aims to formally prove the correct
prediction staying unchanged for any bounded adversarial perturbations.
Our algorithm *β*-CROWN encodes per-neuron split constraints used in branch-and-bound in an
**efficient and highly parallelizable bound propagation** manner (analogous to
[CROWN](https://arxiv.org/pdf/1811.00866.pdf)) **without relying on any
computationally extensive LP solvers**.  Thanks to its efficient bound
propagation procedure, *β*-CROWN can be implemented on GPUs to significantly
speedup branch and bound based formal verification.  Additionally, *β*-CROWN
bounds can be optimized cheaply using gradients obtained by autodiff, allowing
it to obtain tighter bounds than typical LP verifiers when intermediate layers' bounds are
jointly optimized.

Details of *β*-CROWN can be found in our paper:

* [Beta-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints for Complete and Incomplete Neural Network Robustness Verification](https://arxiv.org/pdf/2103.06624.pdf).  
NeurIPS 2021  
**Shiqi Wang**\*, **Huan Zhang**\*, **Kaidi Xu**\*, Suman Jana, Xue Lin, Cho-Jui Hsieh and Zico Kolter  
(\* **Equal contribution**)

*β*-CROWN is very fast and also produces better bounds than existing complete and
incomplete neural network verifiers in several settings:

- For complete verification, we are about **three magnitudes faster** than the
  traditional linear programming (LP) based branch and bound approach on CPU such as [(Ehlers,
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
  error (lower bound) from 33% (SDP-FO) to 6% (*β*-CROWN).


![comparison of bounds for incomplete verification](http://www.huan-zhang.com/images/paper/beta_crown_incomplete.png)

The two figures above show the verification lower bounds (y-axis) computed by
*β*-CROWN (**~1 minute** in average per datapoint),
[SDP-FO](https://github.com/deepmind/jax_verify) (**2 to 3 hours** per
datapoint) and linear programming (LP), as well as the upper bounds (x-axis)
computed by PGD adversarial attacks.  Higher quality bounds are closer to y=x
(black line).  On these adversarially trained models, a typical LP based
verifier gives quite loose bounds. *β*-CROWN gives similar or better
quality of bounds compared to SDP-FO while using only a fraction of time.

## Reproducing Experimental Results

*β*-CROWN is one of the key components in α,β-CROWN (alpha-beta-CROWN), the winning verifier in [VNN-COMP 2021](https://sites.google.com/view/vnn2021).

**[important] Please note that the implementation of *β*-CROWN is now merged into the [official repo of α,β-CROWN](https://github.com/huanzhang12/alpha-beta-CROWN)**. In this repository, we only provide the commands that one needs to reproduce the results in *β*-CROWN paper using the α,β-CROWN verifier. You can find comprehensive [Usage Documentation](https://github.com/huanzhang12/alpha-beta-CROWN/tree/main/docs/usage.md) on how to easily use our α,β-CROWN verifier and [customize it for your own purpose](https://github.com/huanzhang12/alpha-beta-CROWN#run-%CE%B1%CE%B2-crown-alpha-beta-crown-verifier-on-your-own-model) (e.g., models and properties).

### Installation and Setup

Our code is based on Python 3.7+ and PyTorch 1.8.x LTS. It can be installed
easily into a conda environment. If you don't have conda, you can install
[miniconda](https://docs.conda.io/en/latest/miniconda.html).

```bash
# Clone the alpha-beta-CROWN verifier
git clone https://github.com/huanzhang12/alpha-beta-CROWN.git
cd alpha-beta-CROWN
# Remove the old environment, if necessary.
conda deactivate; conda env remove --name alpha-beta-crown
conda env create -f complete_verifier/environment.yml  # install all dependents into the alpha-beta-crown environment
conda activate alpha-beta-crown  # activate the environment
```

To reproduce the results in *β*-CROWN paper, you need the [`robustness_verifier.py`](https://github.com/huanzhang12/alpha-beta-CROWN/tree/main/complete_verifier/robustness_verifier.py) frontend. All parameters for the verifier are defined in a `yaml` config file. For example, to run robustness verification on a CIFAR-10 ResNet network, you just run:

```bash
conda activate alpha-beta-crown  # activate the conda environment
cd complete_verifier
python robustness_verifier.py --config exp_configs/cifar_resnet_2b.yaml
```

You can find explanations for most useful parameters in [this example config
file](https://github.com/huanzhang12/alpha-beta-CROWN/tree/main/complete_verifier/exp_configs/cifar_resnet_2b.yaml). For detailed usage please see the
[Usage Documentation](https://github.com/huanzhang12/alpha-beta-CROWN/tree/main/docs/usage.md).  We also provide a large range of examples in
the [`complete_verifier/exp_configs`](https://github.com/huanzhang12/alpha-beta-CROWN/tree/main/complete_verifier/exp_configs) folder.

### Complete verification experiments

We use the set of oval20 CIFAR-10 models (base, wide and deep) and the corresponding properties provided in [VNN-COMP
2020](https://github.com/verivital/vnn-comp/tree/master/2020/CNN/).  This set
of models and properties have become the standard benchmark in a few papers in complete
verification.

To reproduce our results on CIFAR-10 Base, Wide and Deep model, please run:

```bash
cd complete_verifier
python robustness_verifier.py --config exp_configs/oval_base.yaml  # CIFAR-10 Base
python robustness_verifier.py --config exp_configs/oval_wide.yaml  # CIFAR-10 Wide
python robustness_verifier.py --config exp_configs/oval_deep.yaml  # CIFAR-10 Deep
```

After finishing running the command, you should see the reported mean and median of the running time and the number of branches for all properties in the dataset.

### Incomplete verification experiments

For the incomplete verification setting, we obtain all nine models from the ERAN benchmark
(used in [Singh, et al.](https://openreview.net/pdf?id=S1gParBeIH) and many other papers), including
6 MNIST models and 3 CIFAR-10 models. Moreover, we collect seven 
models from [Dathathri et al. 2020](https://arxiv.org/abs/2010.11645) including
1 MNIST model and 6 CIFAR-10 models which were used to evaluate SDP-FO.
These models were purely PGD adversarially trained (without a certified defense),
and prior to our work, the strong and expensive semidefinite programming based relaxation (which
SDP-FO is based on) was the only possible way to obtain good verified
accuracy.

To reproduce our results for incomplete verification (figures of the verification lower bound), please run: 

```bash
python robustness_verifier.py --config exp_configs/mnist_sample.yaml  # MNIST CNN-A-Adv
python robustness_verifier.py --config exp_configs/cifar_sample.yaml  # CIFAR-10 CNN-B-Adv
```

To reproduce our results for incomplete verification (verified accuracy), please run:

```bash
# ERAN models (convoluational models)
python robustness_verifier.py --config exp_configs/mnist_conv_small.yaml  # MNIST ConvSmall
python robustness_verifier.py --config exp_configs/mnist_conv_big.yaml    # MNIST ConvBig
python robustness_verifier.py --config exp_configs/cifar_conv_small.yaml  # CIFAR ConvSmall
python robustness_verifier.py --config exp_configs/cifar_conv_big.yaml    # CIFAR ConvBig
python robustness_verifier.py --config exp_configs/cifar_resnet.yaml      # CIFAR ResNet

# SDP models
python robustness_verifier.py --config exp_configs/mnist_cnn_a_adv.yaml   # MNIST CNN-A-Adv
python robustness_verifier.py --config exp_configs/cifar_cnn_b_adv.yaml   # CIFAR CNN-B-Adv
python robustness_verifier.py --config exp_configs/cifar_cnn_b_adv4.yaml  # CIFAR CNN-B-Adv-4
python robustness_verifier.py --config exp_configs/cifar_cnn_a_adv.yaml   # CIFAR CNN-A-Adv
python robustness_verifier.py --config exp_configs/cifar_cnn_a_adv4.yaml  # CIFAR CNN-A-Adv-4
python robustness_verifier.py --config exp_configs/cifar_cnn_a_mix.yaml   # CIFAR CNN-A-Mix
python robustness_verifier.py --config exp_configs/cifar_cnn_a_mix4.yaml  # CIFAR CNN-A-Mix-4
```

### Incomplete verification experiments for MLP models

For small MLP models, we find that using MIP to compute the lower and upper bounds for each intermediate layer neuron is quite helpful. Based on the tightened bounds from α-CROWN and MIP, we can further use β-CROWN to do complete verification with BaB. To reproduce the results we reported for MLP models (mnist9_200, mnist9_100, mnist6_200, mnist_6_100) in the ERAN benchmarks, please run:

```bash
# ERAN models (MLP models)
python robustness_verifier.py --config exp_configs/mnist_6_100.yaml  # MNIST MLP 6_100
python robustness_verifier.py --config exp_configs/mnist_6_200.yaml  # MNIST MLP 6_200
python robustness_verifier.py --config exp_configs/mnist_9_100.yaml  # MNIST MLP 9_100
python robustness_verifier.py --config exp_configs/mnist_9_200.yaml  # MNIST MLP 9_200
```

### BibTex Entry

```
@article{wang2021beta,
  title={{Beta-CROWN}: Efficient bound propagation with per-neuron split constraints for complete and incomplete neural network verification},
  author={Wang, Shiqi and Zhang, Huan and Xu, Kaidi and Lin, Xue and Jana, Suman and Hsieh, Cho-Jui and Kolter, J Zico},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```
