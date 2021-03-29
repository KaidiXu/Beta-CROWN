from modules import Flatten
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np

import pandas as pd

########################################
# Defined the model architectures
########################################
def cifar_model():
    # cifar base
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(1024, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def cifar_model_deep():
    # cifar deep
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*8*8, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def cifar_model_wide():
    # cifar wide
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def cnn_4layer():
    # cifar_cnn_a
    return cifar_model_wide()


def cnn_4layer_b():
    # cifar_cnn_b
    return nn.Sequential(
        nn.ZeroPad2d((1,2,1,2)),
        nn.Conv2d(3, 32, (5,5), stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(32, 128, (4,4), stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8192, 250),
        nn.ReLU(),
        nn.Linear(250, 10),
    )


def mnist_cnn_4layer():
    # mnist_cnn_a
    return nn.Sequential(
        nn.Conv2d(1, 16, (4,4), stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, (4,4), stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(1568, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )


def load_model(args, weights_loaded=True):
    """
    Load the model architectures and weights
    """
    model_ori = eval(args.model)()
    print(model_ori)

    if not weights_loaded:
        return model_ori

    map_location = None
    if args.device == 'cpu':
        map_location = torch.device('cpu')

    if 'cnn_4layer' not in args.model:
        model_ori.load_state_dict(torch.load(args.load, map_location)['state_dict'][0])
    else:
        model_ori.load_state_dict(torch.load(args.load, map_location))
        if args.model == "cnn_4layer_b":
            args.MODEL = "b_adv"
            assert args.data == "CIFAR_SAMPLE"
        elif args.model == "cnn_4layer":
            args.MODEL = "a_mix"
            assert args.data == "CIFAR_SAMPLE"
        elif args.model == "mnist_cnn_4layer":
            args.MODEL = "mnist_a_adv"
            assert args.data == "MNIST_SAMPLE"

    return model_ori


########################################
# Preprocess and load the datasets
########################################
def preprocess_cifar(image, inception_preprocess=False, perturbation=False):
    """
    Proprocess images and perturbations.Preprocessing used by the SDP paper.
    """
    MEANS = np.array([125.3, 123.0, 113.9], dtype=np.float32)/255
    STD = np.array([63.0, 62.1, 66.7], dtype=np.float32)/255
    upper_limit, lower_limit = 1., 0.
    if inception_preprocess:
        # Use 2x - 1 to get [-1, 1]-scaled images
        rescaled_devs = 0.5
        rescaled_means = 0.5
    else:
        rescaled_means = MEANS
        rescaled_devs = STD
    if perturbation:
        return image / rescaled_devs
    else:
        return (image - rescaled_means) / rescaled_devs


def load_cifar_sample_data(normalized=True, MODEL="a_mix"):
    """
    Load sampled cifar data: 100 images that are classified correctly by each MODEL
    """
    X = np.load("./data/sample100_unnormalized/"+MODEL+"/X.npy")
    if normalized:
        X = preprocess_cifar(X)
    X = np.transpose(X, (0,3,1,2))
    y = np.load("./data/sample100_unnormalized/"+MODEL+"/y.npy")
    runnerup = np.load("./data/sample100_unnormalized/"+MODEL+"/runnerup.npy")
    X = torch.from_numpy(X.astype(np.float32))
    y = torch.from_numpy(y.astype(np.int))
    runnerup = torch.from_numpy(runnerup.astype(np.int))
    print("############################")
    if normalized:
        print("Sampled data loaded. Data already preprocessed!")
    else:
        print("Sampled data loaded. Data not preprocessed yet!")
    print("Shape:", X.shape, y.shape, runnerup.shape)
    print("X range:", X.max(), X.min(), X.mean())
    print("############################")
    return X, y, runnerup


def load_mnist_sample_data(MODEL="mnist_a_adv"):
    """
    Load sampled mnist data: 100 images that are classified correctly by each MODEL
    """
    X = np.load("./data/sample100_unnormalized/"+MODEL+"/X.npy")
    X = np.transpose(X, (0,3,1,2))
    y = np.load("./data/sample100_unnormalized/"+MODEL+"/y.npy")
    runnerup = np.load("./data/sample100_unnormalized/"+MODEL+"/runnerup.npy")
    X = torch.from_numpy(X.astype(np.float32))
    y = torch.from_numpy(y.astype(np.int))
    runnerup = torch.from_numpy(runnerup.astype(np.int))
    print("############################")
    print("Shape:", X.shape, y.shape, runnerup.shape)
    print("X range:", X.max(), X.min(), X.mean())
    print("############################")
    return X, y, runnerup


def load_dataset(args):
    """
    Load regular data; Robustness region defined in results pickle 
    """
    if args.data == 'MNIST':
        # dummy_input = torch.randn(1, 1, 28, 28)
        test_data = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
        test_data.mean = torch.tensor([0.0])
        test_data.std = torch.tensor([1.0])
        data_max = 1.
        data_min = 0.
    elif args.data == 'CIFAR':
        # dummy_input = torch.randn(1, 3, 32, 32)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
        test_data = datasets.CIFAR10("./data", train=False, download=True,
                                     transform=transforms.Compose([transforms.ToTensor(), normalize]))
        test_data.mean = torch.tensor([0.485, 0.456, 0.406])
        test_data.std = torch.tensor([0.225, 0.225, 0.225])
        # set data_max and data_min to be None if no clip
        data_max = torch.reshape((1. - test_data.mean) / test_data.std, (1, -1, 1, 1))
        data_min = torch.reshape((0. - test_data.mean) / test_data.std, (1, -1, 1, 1))
    return test_data, data_max, data_min


def load_sampled_dataset(args):
    """
    Load sampled data and define the robustness region
    """
    if args.data == "CIFAR_SAMPLE":
        X, labels, runnerup = load_cifar_sample_data(normalized=True, MODEL=args.MODEL)
        data_max = torch.tensor(preprocess_cifar(1.)).reshape(1,-1,1,1)
        data_min = torch.tensor(preprocess_cifar(0.)).reshape(1,-1,1,1)
        eps_temp = 2./255.
        eps_temp = torch.tensor(preprocess_cifar(eps_temp, perturbation=True)).reshape(1,-1,1,1)
    elif args.data == "MNIST_SAMPLE":
        X, labels, runnerup = load_mnist_sample_data(MODEL=args.MODEL)
        data_max = torch.tensor(1.).reshape(1,-1,1,1)
        data_min = torch.tensor(0.).reshape(1,-1,1,1)
        eps_temp = 0.3
        eps_temp = torch.tensor(eps_temp).reshape(1,-1,1,1)
    return X, labels, runnerup, data_max, data_min, eps_temp


########################################
# Load results stored in the pickle files
########################################
def load_pickle_results(args):
    """
    Load results stored in the pickle files
    """
    if 'deep' in args.model:
        gt_results = pd.read_pickle('./data/deep_100.pkl')
    elif 'wide' in args.model:
        gt_results = pd.read_pickle('./data/wide_100.pkl')
    elif args.model == 'cnn_4layer_b':
        gt_results = pd.read_pickle('./data/compare_results/cnn-b-adv.pkl')
        gt_results["Idx"] = np.arange(len(gt_results)).astype(np.int)
    elif args.model == 'cnn_4layer':
        gt_results = pd.read_pickle('./data/compare_results/cnn-a-mix.pkl')
        gt_results["Idx"] = np.arange(len(gt_results)).astype(np.int)
    elif args.model == "mnist_cnn_4layer":
        gt_results = pd.read_pickle('./data/compare_results/mnist-cnn-a-adv.pkl')
        gt_results["Idx"] = np.arange(len(gt_results)).astype(np.int)
    else:
        gt_results = pd.read_pickle('./data/base_100.pkl')
    return gt_results

