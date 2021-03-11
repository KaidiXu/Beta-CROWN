import copy
import time

from torch.nn import ZeroPad2d

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.bound_ops import BoundRelu
from auto_LiRPA.perturbations import *
from modules import Flatten


def simplify_network(all_layers):
    """
    Given a sequence of Pytorch nn.Module `all_layers`,
    representing a feed-forward neural network,
    merge the layers when two sucessive modules are nn.Linear
    and can therefore be equivalenty computed as a single nn.Linear
    """
    new_all_layers = [all_layers[0]]
    for layer in all_layers[1:]:
        if (type(layer) is nn.Linear) and (type(new_all_layers[-1]) is nn.Linear):
            # We can fold together those two layers
            prev_layer = new_all_layers.pop()

            joint_weight = torch.mm(layer.weight.data, prev_layer.weight.data)
            if prev_layer.bias is not None:
                joint_bias = layer.bias.data + torch.mv(layer.weight.data, prev_layer.bias.data)
            else:
                joint_bias = layer.bias.data

            joint_out_features = layer.out_features
            joint_in_features = prev_layer.in_features

            joint_layer = nn.Linear(joint_in_features, joint_out_features)
            joint_layer.bias.data.copy_(joint_bias)
            joint_layer.weight.data.copy_(joint_weight)
            new_all_layers.append(joint_layer)
    return new_all_layers


def add_single_prop(layers, gt, cls):
    """
    gt: ground truth lablel
    cls: class we want to verify against
    """
    additional_lin_layer = nn.Linear(10, 1, bias=True)
    lin_weights = additional_lin_layer.weight.data
    lin_weights.fill_(0)
    lin_bias = additional_lin_layer.bias.data
    lin_bias.fill_(0)
    lin_weights[0, cls] = -1
    lin_weights[0, gt.detach()] = 1

    final_layers = [layers[-1], additional_lin_layer]
    final_layer = simplify_network(final_layers)
    verif_layers = layers[:-1] + final_layer
    for layer in verif_layers:
        for p in layer.parameters():
            p.requires_grad = False

    return verif_layers


class LiRPAConvNet:

    def __init__(self, model_ori, pred, test, solve_slope=False, device='cuda', simplify=True, in_size=(1, 3, 32, 32)):
        """
        convert pytorch model to auto_LiRPA module
        """

        layers = list(model_ori.children())
        if simplify:
            added_prop_layers = add_single_prop(layers, pred, test)
            self.layers = added_prop_layers
        else:
            self.layers = layers
        net = nn.Sequential(*self.layers)
        self.solve_slope = solve_slope
        if solve_slope:
            self.net = BoundedModule(net, torch.rand(in_size), bound_opts={'relu': 'random_evaluation', 'conv_mode': 'patches'},
                                     device=device)
        else:
            self.net = BoundedModule(net, torch.rand(in_size), bound_opts={'relu': 'same-slope'}, device=device)
        self.net.eval()

    def get_lower_bound(self, pre_lbs, pre_ubs, decision, slopes=None,
                        history=[], decision_thresh=0, layer_set_bound=True, beta=True):

        """
        # (in) pre_lbs: layers list -> tensor(batch, layer shape)
        # (in) relu_mask: relu layers list -> tensor(batch, relu layer shape (view-1))
        # (in) slope: relu layers list -> tensor(batch, relu layer shape)
        # (out) lower_bounds: batch list -> layers list -> tensor(layer shape)
        # (out) masks_ret: batch list -> relu layers list -> tensor(relu layer shape)
        # (out) slope: batch list -> relu layers list -> tensor(relu layer shape)
        """
        start = time.time()
        lower_bounds, upper_bounds, masks_ret, slopes = self.update_bounds_parallel(pre_lbs, pre_ubs, decision,
                                                                                    slopes, beta=beta, early_stop=False,
                                                                                    opt_choice="adam", iteration=20,
                                                                                    history=history,
                                                                                    decision_thresh=decision_thresh,
                                                                                    layer_set_bound=layer_set_bound)

        end = time.time()
        print('batch time: ', end - start)
        return [i[-1] for i in upper_bounds], [i[-1] for i in lower_bounds], None, masks_ret, lower_bounds, upper_bounds, slopes

    def get_relu(self, model, idx):
        # find the i-th ReLU layer
        i = 0
        for layer in model.children():
            if isinstance(layer, BoundRelu):
                i += 1
                if i == idx:
                    return layer

    def get_candidate(self, model, lb, ub):
        # get the intermediate bounds in the current model and build self.name_dict which contains the important index
        # and model name pairs

        if self.input_domain.ndim == 2:
            lower_bounds = [self.input_domain[:, 0].squeeze(-1)]
            upper_bounds = [self.input_domain[:, 1].squeeze(-1)]
        else:
            lower_bounds = [self.input_domain[:, :, :, 0].squeeze(-1)]
            upper_bounds = [self.input_domain[:, :, :, 1].squeeze(-1)]
        self.pre_relu_indices = []
        idx, i, model_i = 0, 0, 0
        # build a name_dict to map layer idx in self.layers to BoundedModule
        self.name_dict = {0: model.root_name[0]}
        model_names = list(model._modules)

        for layer in self.layers:
            if isinstance(layer, nn.ReLU):
                i += 1
                this_relu = self.get_relu(model, i)
                lower_bounds[-1] = this_relu.inputs[0].lower.squeeze().detach()
                upper_bounds[-1] = this_relu.inputs[0].upper.squeeze().detach()
                lower_bounds.append(F.relu(lower_bounds[-1]).detach())
                upper_bounds.append(F.relu(upper_bounds[-1]).detach())
                self.pre_relu_indices.append(idx)
                self.name_dict[idx + 1] = model_names[model_i]
                model_i += 1
            elif isinstance(layer, Flatten):
                lower_bounds.append(lower_bounds[-1].reshape(-1).detach())
                upper_bounds.append(upper_bounds[-1].reshape(-1).detach())
                self.name_dict[idx + 1] = model_names[model_i]
                model_i += 8  # Flatten is split to 8 ops in BoundedModule
            elif isinstance(layer, ZeroPad2d):
                lower_bounds.append(F.pad(lower_bounds[-1], layer.padding))
                upper_bounds.append(F.pad(upper_bounds[-1], layer.padding))
                self.name_dict[idx + 1] = model_names[model_i]
                model_i += 24
            else:
                self.name_dict[idx + 1] = model_names[model_i]
                lower_bounds.append([])
                upper_bounds.append([])
                model_i += 1
            idx += 1

        # Also add the bounds on the final thing
        lower_bounds[-1] = (lb.view(-1).detach())
        upper_bounds[-1] = (ub.view(-1).detach())

        return lower_bounds, upper_bounds, self.pre_relu_indices

    def get_candidate_parallel(self, model, lb, ub, batch):
        # get the intermediate bounds in the current model
        lower_bounds = [self.input_domain[:, :, :, 0].squeeze(-1).repeat(batch, 1, 1, 1)]
        upper_bounds = [self.input_domain[:, :, :, 1].squeeze(-1).repeat(batch, 1, 1, 1)]
        idx, i, = 0, 0
        for layer in self.layers:
            if isinstance(layer, nn.ReLU):
                i += 1
                this_relu = self.get_relu(model, i)
                lower_bounds[-1] = this_relu.inputs[0].lower.detach()
                upper_bounds[-1] = this_relu.inputs[0].upper.detach()
                lower_bounds.append(
                    F.relu(lower_bounds[-1]).detach())  # TODO we actually do not need the bounds after ReLU
                upper_bounds.append(F.relu(upper_bounds[-1]).detach())
            elif isinstance(layer, Flatten):
                lower_bounds.append(lower_bounds[-1].reshape(batch, -1).detach())
                upper_bounds.append(upper_bounds[-1].reshape(batch, -1).detach())
            elif isinstance(layer, nn.ZeroPad2d):
                lower_bounds.append(F.pad(lower_bounds[-1], layer.padding).detach())
                upper_bounds.append(F.pad(upper_bounds[-1], layer.padding).detach())

            else:
                lower_bounds.append([])
                upper_bounds.append([])
            idx += 1

        # Also add the bounds on the final thing
        lower_bounds[-1] = (lb.view(batch, -1).detach())
        upper_bounds[-1] = (ub.view(batch, -1).detach())

        return lower_bounds, upper_bounds

    def get_mask_parallel(self, model):
        # get the mask of status of ReLU, 0 means inactive neurons, -1 means unstable neurons, 1 means active neurons
        mask = []
        idx, i, = 0, 0
        for layer in self.layers:
            if isinstance(layer, nn.ReLU):
                i += 1
                this_relu = self.get_relu(model, i)
                mask_tmp = torch.zeros_like(this_relu.inputs[0].lower)
                unstable = ((this_relu.inputs[0].lower < 0) & (this_relu.inputs[0].upper > 0))
                mask_tmp[unstable] = -1
                active = (this_relu.inputs[0].lower >= 0)
                mask_tmp[active] = 1
                # otherwise 0, for inactive neurons

                mask.append(mask_tmp.reshape(mask_tmp.size(0), -1))

        ret = []
        for i in range(mask[0].size(0)):
            ret.append([j[i] for j in mask])

        return ret

    def get_beta(self, model):
        b = []
        bm = []
        for m in model._modules.values():
            if isinstance(m, BoundRelu):
                b.append(m.beta.clone().detach())
                bm.append(m.beta_mask.clone().detach())

        retb = []
        retbm = []
        for i in range(b[0].size(0)):
            retb.append([j[i] for j in b])
            retbm.append([j[i] for j in bm])
        return (retb, retbm)

    def get_slope(self, model):
        s = []
        for m in model._modules.values():
            if isinstance(m, BoundRelu):
                s.append(m.slope.transpose(0, 1).clone().detach())

        ret = []
        for i in range(s[0].size(0)):
            ret.append([j[i] for j in s])
        return ret

    def set_slope(self, model, slope):
        idx = 0
        for m in model._modules.values():
            if isinstance(m, BoundRelu):
                # m.slope = slope[idx].repeat(2, *([1] * (slope[idx].ndim - 1))).requires_grad_(True)
                m.slope = slope[idx].repeat(2, *([1] * (slope[idx].ndim - 1))).transpose(0, 1).requires_grad_(True)
                idx += 1

    def reset_beta(self, model, batch=0):
        if batch == 0:
            for m in model._modules.values():
                if isinstance(m, BoundRelu):
                    m.beta.data = m.beta.data * 0.
                    m.beta_mask.data = m.beta_mask.data * 0.
                    # print("beta[{}]".format(batch), m.beta.shape, m.beta_mask.shape)
        else:
            for m in model._modules.values():
                if isinstance(m, BoundRelu):
                    ndim = m.beta.data.ndim
                    # m.beta.data=(m.beta.data[0:1]*0.).repeat(batch*2, *([1] * (ndim - 1))).requires_grad_(True)
                    # m.beta_mask.data=(m.beta_mask.data[0:1]*0.).repeat(batch*2, *([1] * (ndim - 1))).requires_grad_(True)
                    m.beta = torch.zeros(m.beta[:, 0:1].shape).repeat(1, batch * 2, *([1] * (ndim - 2))).detach().to(
                        m.beta.device).requires_grad_(True)
                    m.beta_mask = torch.zeros(m.beta_mask[0:1].shape).repeat(batch * 2,
                                                                             *([1] * (ndim - 2))).detach().to(
                        m.beta.device).requires_grad_(False)
                    # print("beta[{}]".format(batch), m.beta.shape, m.beta_mask.shape)

    def update_bounds_parallel(self, pre_lb_all=None, pre_ub_all=None, decision=None, slopes=None,
                               beta=True, early_stop=True, opt_choice="default", iteration=20, history=[],
                               decision_thresh=0, layer_set_bound=True):
        # update optimize-CROWN bounds in a parallel way
        total_batch = len(decision)
        decision = np.array(decision)

        layers_need_change = np.unique(decision[:, 0])
        layers_need_change.sort()

        # initial results with empty list
        ret_l = [[] for _ in range(len(decision) * 2)]
        ret_u = [[] for _ in range(len(decision) * 2)]
        masks = [[] for _ in range(len(decision) * 2)]
        ret_s = [[] for _ in range(len(decision) * 2)]

        pre_lb_all_cp = copy.deepcopy(pre_lb_all)
        pre_ub_all_cp = copy.deepcopy(pre_ub_all)

        for idx in layers_need_change:
            # iteratively change upper and lower bound from former to later layer
            tmp_d = np.argwhere(decision[:, 0] == idx)  # .squeeze()
            # idx is the index of relu layers, change_idx is the index of all layers
            change_idx = self.pre_relu_indices[idx]

            batch = len(tmp_d)
            select_history = [history[idx] for idx in tmp_d.squeeze().reshape(-1)]

            if beta:
                # update beta mask, put it after reset_beta
                # reset beta according to the shape of batch
                self.reset_beta(self.net, batch)

                # print("select history", select_history)

                bound_relus = []
                for m in self.net._modules.values():
                    if isinstance(m, BoundRelu):
                        bound_relus.append(m)
                        m.beta_mask.data = m.beta_mask.data.view(batch * 2, -1)

                for bi in range(batch):
                    d = tmp_d[bi][0]
                    # assign current decision to each point of a batch
                    bound_relus[int(decision[d][0])].beta_mask.data[bi, int(decision[d][1])] = 1
                    bound_relus[int(decision[d][0])].beta_mask.data[bi + batch, int(decision[d][1])] = -1
                    # print("assign", bi, decision[d], 1, bound_relus[decision[d][0]].beta_mask.data[bi, decision[d][1]])
                    # print("assign", bi+batch, decision[d], -1, bound_relus[decision[d][0]].beta_mask.data[bi+batch, decision[d][1]])
                    # assign history decision according to select_history
                    for (hid, hl), hc in select_history[bi]:
                        bound_relus[hid].beta_mask.data[bi, hl] = int((hc - 0.5) * 2)
                        bound_relus[hid].beta_mask.data[bi + batch, hl] = int((hc - 0.5) * 2)
                        # print("assign", bi, [hid, hl], hc, bound_relus[hid].beta_mask.data[bi, hl])
                        # print("assign", bi+batch, [hid, hl], hc, bound_relus[hid].beta_mask.data[bi+batch, hl])

                # sanity check: beta_mask should only be assigned for split nodes
                for m in bound_relus:
                    m.beta_mask.data = m.beta_mask.data.view(m.beta[0].shape)

            slope_select = [i[tmp_d.squeeze()].clone() for i in slopes]

            pre_lb_all = [i[tmp_d.squeeze()].clone() for i in pre_lb_all_cp]
            pre_ub_all = [i[tmp_d.squeeze()].clone() for i in pre_ub_all_cp]

            if batch == 1:
                pre_lb_all = [i.clone().unsqueeze(0) for i in pre_lb_all]
                pre_ub_all = [i.clone().unsqueeze(0) for i in pre_ub_all]
                slope_select = [i.clone().unsqueeze(0) for i in slope_select]

            upper_bounds = [i.clone() for i in pre_ub_all[:change_idx + 1]]
            lower_bounds = [i.clone() for i in pre_lb_all[:change_idx + 1]]
            upper_bounds_cp = copy.deepcopy(upper_bounds)
            lower_bounds_cp = copy.deepcopy(lower_bounds)

            for i in range(batch):
                d = tmp_d[i][0]
                upper_bounds[change_idx].view(batch, -1)[i][decision[d][1]] = 0
                lower_bounds[change_idx].view(batch, -1)[i][decision[d][1]] = 0

            pre_lb_all = [torch.cat(2 * [i]) for i in pre_lb_all]
            pre_ub_all = [torch.cat(2 * [i]) for i in pre_ub_all]

            # merge the inactive and active splits together
            new_candidate = {}
            for i, (l, uc, lc, u) in enumerate(zip(lower_bounds, upper_bounds_cp, lower_bounds_cp, upper_bounds)):
                # we set lower = 0 in first half batch, and upper = 0 in second half batch
                new_candidate[self.name_dict[i]] = [torch.cat((l, lc), dim=0), torch.cat((uc, u), dim=0)]

            if not layer_set_bound:
                new_candidate_p = {}
                for i, (l, u) in enumerate(zip(pre_lb_all[:-2], pre_ub_all[:-2])):
                    # we set lower = 0 in first half batch, and upper = 0 in second half batch
                    new_candidate_p[self.name_dict[i]] = [l, u]

            # create new_x here since batch may change
            ptb = PerturbationLpNorm(norm=self.x.ptb.norm, eps=self.x.ptb.eps,
                                     x_L=self.x.ptb.x_L.repeat(batch * 2, 1, 1, 1),
                                     x_U=self.x.ptb.x_U.repeat(batch * 2, 1, 1, 1))
            new_x = BoundedTensor(self.x.data.repeat(batch * 2, 1, 1, 1), ptb)
            self.net(new_x)  # batch may change, so we need to do forward to set some shapes here

            if len(slope_select) > 0:
                # set slope here again
                self.set_slope(self.net, slope_select)

            torch.cuda.empty_cache()
            if layer_set_bound:
                # we fix the intermediate bounds before change_idx-th layer by using normal CROWN
                if self.solve_slope and change_idx >= self.pre_relu_indices[-1]:
                    # we split the ReLU at last layer, directly use Optimized CROWN
                    self.net.set_bound_opts(
                        {'ob_start_idx': sum(change_idx <= x for x in self.pre_relu_indices), 'ob_beta': beta,
                         'ob_update_by_layer': layer_set_bound, 'ob_iteration': iteration})
                    lb, ub, = self.net.compute_bounds(x=(new_x,), IBP=False, C=None, method='CROWN-Optimized',
                                                      new_interval=new_candidate, return_A=False, bound_upper=False)
                else:
                    # we split the ReLU before the last layer, calculate intermediate bounds by using normal CROWN
                    self.net.set_relu_used_count(sum(change_idx <= x for x in self.pre_relu_indices))
                    with torch.no_grad():
                        lb, ub, = self.net.compute_bounds(x=(new_x,), IBP=False, C=None, method='backward',
                                                          new_interval=new_candidate, bound_upper=False, return_A=False)

                # we don't care about the upper bound of the last layer
                lower_bounds_new, upper_bounds_new = self.get_candidate_parallel(self.net, lb, lb + 99, batch * 2)

                if change_idx < self.pre_relu_indices[-1]:
                    # check whether we have a better bounds before, and preset all intermediate bounds
                    for i, (l, u) in enumerate(
                            zip(lower_bounds_new[change_idx + 2:-1], upper_bounds_new[change_idx + 2:-1])):
                        new_candidate[self.name_dict[i + change_idx + 2]] = [
                            torch.max(l, pre_lb_all[i + change_idx + 2]), torch.min(u, pre_ub_all[i + change_idx + 2])]

                    if self.solve_slope:
                        self.net.set_bound_opts(
                            {'ob_start_idx': sum(change_idx <= x for x in self.pre_relu_indices), 'ob_beta': beta,
                             'ob_update_by_layer': layer_set_bound, 'ob_iteration': iteration})
                        lb, ub, = self.net.compute_bounds(x=(new_x,), IBP=False, C=None, method='CROWN-Optimized',
                                                          new_interval=new_candidate, return_A=False, bound_upper=False)
                    else:
                        self.net.set_relu_used_count(sum(change_idx <= x for x in self.pre_relu_indices))
                        with torch.no_grad():
                            lb, ub, = self.net.compute_bounds(x=(new_x,), IBP=False, C=None, method='backward',
                                                              new_interval=new_candidate, bound_upper=False,
                                                              return_A=False)

            else:
                # all intermediate bounds are re-calculate by optimized CROWN
                self.net.set_bound_opts({'ob_start_idx': 99, 'ob_beta': beta, 'ob_update_by_layer': layer_set_bound,
                                         'ob_iteration': iteration})
                lb, ub, = self.net.compute_bounds(x=(new_x,), IBP=False, C=None, method='CROWN-Optimized',
                                                  new_interval=new_candidate_p, return_A=False, bound_upper=False)

            # print('best results of parent nodes', pre_lb_all[-1].repeat(2, 1))
            # print('finally, after optimization:', lower_bounds_new[-1])

            # primal = self.get_primals(A_dict, return_x=True)
            lower_bounds_new, upper_bounds_new = self.get_candidate_parallel(self.net, lb, lb + 99, batch * 2)

            lower_bounds_new[-1] = torch.max(lower_bounds_new[-1], pre_lb_all[-1])
            upper_bounds_new[-1] = torch.min(upper_bounds_new[-1], pre_ub_all[-1])

            mask = self.get_mask_parallel(self.net)
            if len(slope_select) > 0:
                slope = self.get_slope(self.net)

            # reshape the results
            for i in range(len(tmp_d)):
                ret_l[int(tmp_d[i])] = [j[i] for j in lower_bounds_new]
                ret_l[int(tmp_d[i] + total_batch)] = [j[i + batch] for j in lower_bounds_new]

                ret_u[int(tmp_d[i])] = [j[i] for j in upper_bounds_new]
                ret_u[int(tmp_d[i] + total_batch)] = [j[i + batch] for j in upper_bounds_new]

                masks[int(tmp_d[i])] = mask[i]
                masks[int(tmp_d[i] + total_batch)] = mask[i + batch]
                if len(slope_select) > 0:
                    ret_s[int(tmp_d[i])] = slope[i]
                    ret_s[int(tmp_d[i] + total_batch)] = slope[i + batch]

        return ret_l, ret_u, masks, ret_s

    def fake_forward(self, x):
        for layer in self.layers:
            if type(layer) is nn.Linear:
                x = F.linear(x, layer.weight, layer.bias)
            elif type(layer) is nn.Conv2d:
                x = F.conv2d(x, layer.weight, layer.bias, layer.stride, layer.padding, layer.dilation, layer.groups)
            elif type(layer) == nn.ReLU:
                x = F.relu(x)
            elif type(layer) == Flatten:
                x = x.reshape(x.shape[0], -1)
            elif type(layer) == nn.ZeroPad2d:
                x = F.pad(x, layer.padding)
            else:
                print(type(layer))
                raise NotImplementedError

        return x

    def get_primals(self, A, return_x=False):
        # get primal input by using A matrix
        input_A_lower = A[self.layer_names[-1]][self.net.input_name[0]][0]
        batch = input_A_lower.shape[1]
        l = self.input_domain[:, :, :, 0].repeat(batch, 1, 1, 1)
        u = self.input_domain[:, :, :, 1].repeat(batch, 1, 1, 1)
        diff = 0.5 * (l - u)  # already flip the sign by using lower - upper
        net_input = diff * torch.sign(input_A_lower.squeeze(0)) + self.x
        if return_x: return net_input

        primals = [net_input]
        for layer in self.layers:
            if type(layer) is nn.Linear:
                pre = primals[-1]
                primals.append(F.linear(pre, layer.weight, layer.bias))
            elif type(layer) is nn.Conv2d:
                pre = primals[-1]
                primals.append(F.conv2d(pre, layer.weight, layer.bias,
                                        layer.stride, layer.padding, layer.dilation, layer.groups))
            elif type(layer) == nn.ReLU:
                primals.append(F.relu(primals[-1]))
            elif type(layer) == Flatten:
                primals.append(primals[-1].reshape(primals[-1].shape[0], -1))
            else:
                print(type(layer))
                raise NotImplementedError

        # primals = primals[1:]
        primals = [i.detach().clone() for i in primals]
        # print('primals', primals[-1])

        return net_input, primals

    def get_relu_mask(self):
        relu_mask = []
        relu_idx = 0
        for layer in self.layers:
            if type(layer) == nn.ReLU:
                relu_idx += 1
                this_relu = self.get_relu(self.net, relu_idx)
                new_layer_mask = []
                ratios_all = this_relu.d.squeeze(0)
                for slope in ratios_all.flatten():
                    if slope.item() == 1.0:
                        new_layer_mask.append(1)
                    elif slope.item() == 0.0:
                        new_layer_mask.append(0)
                    else:
                        new_layer_mask.append(-1)
                relu_mask.append(torch.tensor(new_layer_mask).to(self.x.device))

        return relu_mask

    def build_the_model(self, input_domain, x, no_lp=True, decision_thresh=0):
        self.x = x
        self.input_domain = input_domain

        slope_opt = None

        # first get CROWN bounds
        if self.solve_slope:
            self.net.init_slope(self.x)
            self.net.set_bound_opts({'ob_iteration': 100, 'ob_beta': False, 'ob_alpha': True, 'ob_opt_choice': "adam",
                                     'ob_decision_thresh': decision_thresh, 'ob_early_stop': False, 'ob_log': False,
                                     'ob_start_idx': 99, 'ob_keep_best': True, 'ob_update_by_layer': True,
                                     'ob_lr': 0.1})
            lb, ub, A_dict = self.net.compute_bounds(x=(x,), IBP=False, C=None, method='CROWN-Optimized', return_A=True,
                                                     bound_upper=False)
            slope_opt = self.get_slope(self.net)[0]  # initial with one node only
        else:
            with torch.no_grad():
                lb, ub, A_dict = self.net.compute_bounds(x=(x,), IBP=False, C=None, method='backward', return_A=True)

        # build a complete A_dict
        self.layer_names = list(A_dict[list(A_dict.keys())[-1]].keys())[2:]
        self.layer_names.sort()

        # update bounds
        print('initial CROWN bounds:', lb, ub)
        primals, mini_inp = None, None
        # mini_inp, primals = self.get_primals(self.A_dict)
        lb, ub, pre_relu_indices = self.get_candidate(self.net, lb, lb + 99)  # primals are better upper bounds
        duals = None

        return ub[-1], lb[-1], mini_inp, duals, primals, self.get_relu_mask(), lb, ub, pre_relu_indices, slope_opt
