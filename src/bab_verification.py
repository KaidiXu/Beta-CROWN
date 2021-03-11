import argparse
import copy
import random
import sys
import time
import gc
sys.path.append('../auto_LiRPA')

from model_beta_CROWN import LiRPAConvNet
from relu_conv_parallel import relu_bab_parallel

from utils import *

import numpy as np
import json

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *


parser = argparse.ArgumentParser()

parser.add_argument('--no_solve_slope', action='store_false', dest='solve_slope', help='do not optimize slope/alpha in compute bounds')
parser.add_argument("--load", type=str, default="sdp_models/cnn_b_adv.model", help='Load pretrained model')
parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help='use cpu or cuda')
parser.add_argument("--data", type=str, default="CIFAR_SAMPLE", choices=["MNIST", "CIFAR", "CIFAR_SAMPLE", "MNIST_SAMPLE"], help='dataset')
parser.add_argument("--seed", type=int, default=100, help='random seed')
parser.add_argument("--norm", type=float, default='inf', help='p norm for epsilon perturbation')
parser.add_argument("--bound_type", type=str, default="CROWN-IBP",
                    choices=["IBP", "CROWN-IBP", "CROWN"], help='method of bound analysis')
parser.add_argument("--model", type=str, default="cnn_4layer_b",
                    help='model name (cifar_model, cifar_model_deep, cifar_model_wide, cnn_4layer, cnn_4layer_b, mnist_cnn_4layer)')
parser.add_argument("--batch_size", type=int, default=64, help='batch size')
parser.add_argument("--bound_opts", type=str, default="same-slope", choices=["same-slope", "zero-lb", "one-lb"],
                    help='bound options for relu')
parser.add_argument('--no_warm', action='store_true', default=False, help='using warm up for lp solver, true by default')
parser.add_argument('--no_beta', action='store_true', default=False, help='using beta splits, true by default')
parser.add_argument("--max_subproblems_list", type=int, default=200000, help='max length of sub-problems list')
parser.add_argument("--decision_thresh", type=float, default=0, help='decision threshold of lower bounds')
parser.add_argument("--timeout", type=int, default=3600, help='timeout for one property')
parser.add_argument("--start", type=int, default=0, help='start from i-th property')
parser.add_argument("--end", type=int, default=100, help='end with (i-1)-th property')
parser.add_argument("--mode", type=str, default="complete", choices=["complete", "incomplete", "verified-acc"], help='which mode to use')

args = parser.parse_args()

def bab(model_ori, data, target, norm, eps, args, data_max=None, data_min=None):

    if norm == np.inf:
        if data_max is None:
            # data_max = torch.reshape((1. - loader.mean) / loader.std, (1, -1, 1, 1))
            # data_min = torch.reshape((0. - loader.mean) / loader.std, (1, -1, 1, 1))
            data_ub = data + eps  # torch.min(data + eps, data_max)  # eps is already normalized
            data_lb = data - eps  # torch.max(data - eps, data_min)
        else:
            data_ub = torch.min(data + eps, data_max)
            data_lb = torch.max(data - eps, data_min)
    else:
        data_ub = data_lb = data

    pred = torch.argmax(model_ori(data), dim=1)
    # LiRPA wrapper
    model = LiRPAConvNet(model_ori, pred, target, solve_slope=args.solve_slope, device=args.device, in_size=data.shape)

    if list(model.net.parameters())[0].is_cuda:
        data = data.cuda()
        data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

    ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb).to(data_lb.device)
    domain = torch.stack([data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1)

    # with torch.autograd.set_detect_anomaly(True):
    print('beta splits:', not args.no_beta)
    min_lb, min_ub, ub_point, nb_states = relu_bab_parallel(model, domain, x, batch=args.batch_size, no_LP=True,
                                                            decision_thresh=args.decision_thresh, beta=not args.no_beta,
                                                            max_subproblems_list=args.max_subproblems_list,
                                                            timeout=args.timeout)

    if isinstance(min_lb, torch.Tensor):
        min_lb = min_lb.item()
    return min_lb, nb_states


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == "incomplete":
        print("incomplete verification, set decision_thresh to be inf") 
        args.decision_thresh = float('inf')
    elif args.mode == "verified-acc":
        print("complete verification for verified accuracy, set decision_thresh to be 0") 
        # args.decision_thresh = float('inf')
        args.decision_thresh = 0

    model_ori = load_model(args, weights_loaded=True)

    if "SAMPLE" in args.data:
        X, labels, runnerup, data_max, data_min, eps_temp = load_sampled_dataset(args)
    else:
        test_data, data_max, data_min = load_dataset(args)

    # loading verification properties from pickle files
    gt_results = load_pickle_results(args)

    bnb_ids = gt_results.index[args.start:args.end]

    ret = []
    verified_acc = len(bnb_ids)
    for new_idx, idx in enumerate(bnb_ids):
        print('\n %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% idx:', new_idx, 'img ID:', int(gt_results.loc[idx]["Idx"]), '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        torch.cuda.empty_cache()
        print(gt_results.loc[idx])

        if "SAMPLE" not in args.data:
            imag_idx = int(gt_results.loc[idx]["Idx"])
            prop_idx = int(gt_results.loc[idx]['prop'])
            eps_temp = gt_results.loc[idx]["Eps"]  # the eps_temp here is already normalized
            x, y = test_data[imag_idx]
        else:
            imag_idx = int(gt_results.loc[idx]["Idx"])
            prop_idx = int(gt_results.loc[idx]["target"])
            label_idx = int(gt_results.loc[idx]["label"])
            # x, y = X[imag_idx], label_idx
            assert prop_idx == runnerup[imag_idx] and label_idx == labels[imag_idx],\
                            "wrong pickle file or sampled dataset"
            x, y, prop_idx = X[imag_idx], labels[imag_idx].item(), runnerup[imag_idx].item()

        x = x.unsqueeze(0)
        model_ori.to('cpu')
        # first check the model is correct at the input
        y_pred = torch.max(model_ori(x)[0], 0)[1].item()

        print('predicted label ', y_pred, ' correct label ', y)
        if y_pred != y:
            print('model prediction is incorrect for the given model')
            verified_acc -= 1
            continue
        if prop_idx is None:
            choices = list(range(10))
            choices.remove(y_pred)
            prop_idx = random.choice(choices)
            print(f"no prop_idx is given, randomly select one")

        if args.mode == "verified-acc":
            pidx_start, pidx_end = 0, 10
            save_path = 'Verified-acc_{}_{}_alpha01_beta_{}_005_iter20_b{}_start_{}.npy'.\
                        format(args.model, args.data, not args.no_beta, args.batch_size, args.start)
        else:
            pidx_start, pidx_end = prop_idx, prop_idx+1
            save_path = '{}_{}_alpha01_beta_{}_005_iter20_b{}_start_{}.npy'.\
                        format(args.model, args.data, not args.no_beta, args.batch_size, args.start)

        pidx_all_verified = True
        for pidx in range(pidx_start, pidx_end):
            print('##### [{}:{}] Tested against {} ######'.format(new_idx, imag_idx, pidx))
            torch.cuda.empty_cache()
            gc.collect()

            model_ori.to('cpu')

            if pidx == y:
                print("correct label, skip!")
                ret.append([imag_idx, 0, 0, 0, new_idx, pidx])
                continue

            start = time.time()
            try:
                # if "SAMPLE" in args.data and args.mode == "incomplete" and pidx == runnerup[imag_idx].item():
                #     # One can slightly improve the results by using PGD bounds as decision_thresh, we disable it by default
                #     args.decision_thresh = gt_results.loc[idx]["pgd lower bound"]

                # Main function to run verification
                l, nodes = bab(model_ori, x, pidx, args.norm, eps_temp, args, 
                                data_max=data_max, data_min=data_min)

                if "SAMPLE" in args.data and args.mode == "incomplete" and pidx == runnerup[imag_idx].item():
                    # PGD should always be larger than l for prop_idx incomplete setting
                    assert gt_results.loc[idx]["pgd lower bound"] > l, "pgd: {} vs l:{}".\
                                    format(gt_results.loc[idx]["pgd lower bound"], l)

                print('Image {} verify end, Time cost: {}'.format(new_idx, time.time()-start))
                ret.append([imag_idx, l, nodes, time.time()-start, new_idx, pidx])
                print(gt_results.loc[idx], l)
                np.save(save_path, np.array(ret))
                if l < 0:
                    pidx_all_verified = False
            except KeyboardInterrupt:
                print('time:', imag_idx, time.time()-start, "\n",)
                print(ret)
                break
            print(ret)
            # break
        if not pidx_all_verified: 
            verified_acc -= 1

    # some results analysis
    np.set_printoptions(suppress=True)
    ret = np.array(ret)
    print(ret)
    print('time mean: {}, branches mean: {}, number of timeout: {}'.\
          format(ret[:, 3].mean(), ret[:, 2].mean(), (ret[:, 1] < 0).sum()))

    print('time median: {}, branches median: {}, number of timeout: {}'.
          format(np.median(ret[:, 3]), np.median(ret[:, 2]), (ret[:, 1] < 0).sum()))

    if args.mode == "verified-acc":
        print("final verified acc: {}%[{}]".format(verified_acc/len(bnb_ids)*100., len(bnb_ids)))


if __name__ == "__main__":
    main(args)
