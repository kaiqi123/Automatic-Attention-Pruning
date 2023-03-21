import math
import random
from itertools import repeat
import numpy as np
import torch
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn as nn
import os
import sys
import json
from pruning import pruning_utils


def save_models_for_inference_time(_name, _module, prune_index, pruning_round, SAVE_NAME, power_value=-1):
    assert SAVE_NAME != ""
    assert pruning_round != -1
    save_dir = f"./inferenece_time_measure/model_results/{SAVE_NAME}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    org_filters_num = _module.out_features if "fc" in _name else _module.out_channels
    
    save_content={"layer_name": _name, "org_filters_num": org_filters_num, \
        "remain_num_filters": org_filters_num-len(prune_index), "prune_index": prune_index.tolist(), "p": power_value}
    with open(save_dir+f"/round_{pruning_round}.json", "a") as f:
        json.dump(save_content, f)
        f.write("\n")


class PruningModule(Module):
    def prune_next_conv(self, curr_name, prune_index, resample, reinit):
        next_conv_dict = {}
        layer_name_list = [layer_name for layer_name, _ in self.named_modules() if "conv" in layer_name and "relu" not in layer_name]
        for i in range(0, len(layer_name_list)-1):
            next_conv_dict[layer_name_list[i]] = layer_name_list[i+1]

        for _name, _module in self.named_modules():
            if curr_name in next_conv_dict.keys() and next_conv_dict[curr_name] == _name:
                print(f"curr_name: {curr_name}, next layer name (_name): {_name}")        
                _module.mask_on_input_channels(prune_index=prune_index, resample=resample, reinit=reinit)


    def prune_filters_by_AAP(
        self, 
        resample=False, 
        reinit=False, 
        SAVE_NAME="", 
        pruning_round=-1, 
        args="",
        conv_threshold=-1,
        power_value=1,
        ):
        
        # calculate total_alive_relu, then calculate std and mean of total_alive_relu
        print("1. Calculate total_alive_relu, then calculate std and mean of total_alive_relu")
        total_alive_relu = []
        for name, p in self.named_parameters():
            # We do not prune bias term, and fc3
            if 'bias' in name or 'mask' in name or "fc3" in name or "bn" in name or "downsample" in name or "relu_para" in name or "shortcut" in name:
                continue
            
            weights_arr = p.data.cpu().numpy() # conv: (out_channel, in_channel, kernal_size, kernal_size)
            
            # calculate the abs sum of each filter, filers_sum is 1-d array
            if "conv" in name:
                weights_list = list(abs(weights_arr))
                filers_sum = np.array([np.sum(e) for e in weights_list]) # dimension: (out_channel, )
            else:
                raise ValueError(name)

            # find its alive relu_arr
            for module_name, module in self.named_modules():
                if module_name == name.replace(f'.{name.split(".")[-1]}','') + "_relu":
                    
                    relu_arr = module.output.data.cpu().numpy() # dimension: fc_relu (bs, out_channel), conv_relu (bs, out_channel, image_size, image_size)
                    assert relu_arr.shape[1] == weights_arr.shape[0]
           
                    # calculating std
                    # relu_arr: (bs, out_channel, image_size, image_size)
                    relu_arr_mean_3d = np.mean(relu_arr, axis=0) # (out_channel, image_size, image_size)
                    alive_index = list(np.nonzero(filers_sum)[0])
                    img_arr_org = relu_arr_mean_3d
                    mean_list = [] # store the mean value of the activations of each filter
                    for num_channel in range(0, img_arr_org.shape[0]):
                        img_arr = img_arr_org[num_channel] # image for one filter: (image_size, image_size)
                        img_arr_mean = np.mean(img_arr)
                        mean_list.append(img_arr_mean)
                    assert len(mean_list) == img_arr_org.shape[0]
                    alive_mean_list = list(np.array(mean_list)[alive_index])
                    total_alive_relu = total_alive_relu + alive_mean_list
        
        # calculate std according to total_alive_relu
        std_total_alive_relu = np.std(total_alive_relu, dtype=np.float64)
        mean_total_alive_relu = np.mean(total_alive_relu)
        
        print("=====================>conv_threshold: {}".format(conv_threshold))

        # Calculate weights of each layer according to its number of parameters or flops
        # Remember to change to following 2 lines:
        # curr_conv_threshold = conv_threshold * threshold_weights_dict_para[name] # set threshold using weights calculated by parameters
        # curr_conv_threshold = conv_threshold * threshold_weights_dict_flops[name] # set threshold using weights calculated by flops
        print("2. Calculate weights of each layer according to its number of parameters")
        para_total_nz = 0.0
        flops_total_nz = 0.0
        threshold_weights_dict_para = {}
        threshold_weights_dict_flops = {}
        for name, module in self.named_modules():
            if "relu" in name and "conv" in name:
                # find relu's conresponding conv layer
                for parameter_name, p in self.named_parameters():
                    if "conv" in parameter_name and "weight_prune" in parameter_name:
                        if name.split("_")[0] == parameter_name.replace(f'.{parameter_name.split(".")[-1]}',''):
                            # calculate parameters
                            tensor = p.data.cpu().numpy()
                            para_curr_nz = np.count_nonzero(tensor)
                            para_total_nz = para_total_nz + para_curr_nz
                            threshold_weights_dict_para[name]=para_curr_nz
                            
                            # calculate flops
                            if "/tiny-imagenet-200" == args.data:
                                if 'vgg' in args.arch:
                                    image_size = pruning_utils.calculate_image_size_tinyImagenet_vgg(parameter_name)
                                elif 'resnet101' == args.arch:
                                    image_size = pruning_utils.calculate_image_size_resnet_bottlenet(parameter_name, image_size=64)
                                else:
                                    raise EOFError(f"On tiny-imagenet-200, {args.arch} is not implemented!")  
                            elif "cifar10" in args.data:
                                if args.arch in ['resnet18Cifar', 'resnet50Cifar', 'resnet56Cifar']:
                                    image_size = pruning_utils.calculate_image_size_cifar10_resnet_basicblock(parameter_name)
                                elif 'resnet101' == args.arch:
                                    image_size = pruning_utils.calculate_image_size_resnet_bottlenet(parameter_name, image_size=32)
                                elif 'vgg' in args.arch:
                                    image_size = pruning_utils.calculate_image_size_cifar10_vgg(parameter_name)
                                elif 'mobilenetV2' == args.arch:
                                    image_size = pruning_utils.calculate_image_size_cifar10_mobilenet(parameter_name)
                                elif 'shufflenetV2' == args.arch:
                                    image_size = pruning_utils.calculate_image_size_cifar10_shufflenet(parameter_name)
                                else:
                                    raise EOFError(f"On cifar10, {args.arch} is not implemented!")
                            else:
                                raise EOFError(f"{args.data} is not implemented!")
                            flops_curr_nz = para_curr_nz * 2 * image_size * image_size
                            flops_total_nz = flops_total_nz + flops_curr_nz
                            threshold_weights_dict_flops[name] = flops_curr_nz
    
        for k,v in threshold_weights_dict_para.items():
            threshold_weights_dict_para[k] = v / para_total_nz
        for k,v in threshold_weights_dict_flops.items():
            threshold_weights_dict_flops[k] = v / flops_total_nz

        assert round(sum(threshold_weights_dict_para.values()), 2) == 1.0
        assert round(sum(threshold_weights_dict_flops.values()), 2) == 1.0

        # prune filters in each layer
        print("3. Prune filters in each layer")
        save_content_threshold=f"pruning_round = {pruning_round}\t"
        for name, module in self.named_modules():

            if "relu" in name and "conv" in name:
                relu_arr = module.output.data.cpu().numpy()

                relu_arr_mean = np.array([np.mean(np.power(filter_relu, power_value)) for filter_relu in list(np.mean(relu_arr, axis=0))])
                
                assert name in threshold_weights_dict_para.keys() 
                assert name in threshold_weights_dict_flops.keys() 
                curr_conv_threshold = conv_threshold * threshold_weights_dict_para[name] # set threshold using weights calculated by parameters
                # curr_conv_threshold = conv_threshold * threshold_weights_dict_flops[name] # set threshold using weights calculated by flops
                prune_index = np.where(relu_arr_mean <= curr_conv_threshold)[0] 
                print(f"curr_conv_threshold: {curr_conv_threshold}")
        
                print(f"conv_threshold: {conv_threshold}, power_value: {power_value}")
                print("relu_arr_mean.shape: {}, average of relu_arr_mean: {}".format(relu_arr_mean.shape, np.mean(relu_arr_mean)))

                # save relu_arr_mean of current layer
                metrics_relu_arr_mean = [np.std(relu_arr_mean, dtype=np.float64), np.mean(relu_arr_mean), np.min(relu_arr_mean), np.max(relu_arr_mean)]
                save_content_threshold = save_content_threshold + f"{name}={metrics_relu_arr_mean}\t"

                print("====>Begin to find the next conv module, and put mask on input channel")
                self.prune_next_conv(
                    curr_name=name.split("_")[0], 
                    prune_index=prune_index, 
                    resample=resample, 
                    reinit=reinit)

                print("====>Begin to find the fc or conv module, and prune filters (put mask on output channel)")
                for _name, _module in self.named_modules():
                    if name.split("_")[0] == _name:
                        print(_name, _module)
                        _module.prune_filters(prune_index=prune_index, resample=resample, reinit=reinit)

                        # calculate remaining filters to test inference time
                        print("=====>Save number of remaining filters for inference time")
                        assert pruning_round != -1
                        save_models_for_inference_time(_name, _module, prune_index, pruning_round, SAVE_NAME, power_value=-1)
        
        # save threhold of each layer
        with open(args.workspace+"/threshold_per_layer.txt", "a") as f:
            f.write(save_content_threshold+"\n")

        metric_list = [std_total_alive_relu, mean_total_alive_relu]
        return metric_list


class MaskedLinear(Module):
    r"""Applies a masked linear transformation to the incoming data: :math:`y = (A * M)x + b`
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``
    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.
    Attributes:
        weight: the learnable weights of the module of shape
            (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)
        mask: the unlearnable mask for the weight.
            It has the same shape as weight (out_features x in_features)
    """
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        # Initialize the mask with 1
        self.mask = Parameter(torch.ones([out_features, in_features]), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
            The resulting tensor will have values sampled from U(−bound,bound)
            a – the negative slope of the rectifier used after this layer (only with 'leaky_relu') (used)
            fan_in preserves the magnitude of the variance of the weights in the forward pass
        """
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight * self.mask, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'

    def prune(self, threshold, resample, reinit=False):
        print("fc, prune weights, percentile_value: {}".format(threshold))
        weight_dev = self.weight.device
        mask_dev = self.mask.device

        # Convert Tensors to numpy and calculate
        tensor = self.weight.data.cpu().numpy()
        mask = self.mask.data.cpu().numpy()
        new_mask = np.where(abs(tensor) < threshold, 0, mask)
        if resample:
            new_mask = np.random.permutation(new_mask)
        
        # Apply new weight and mask
        self.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)

        if reinit:
            nn.init.xavier_uniform_(self.weight)
            self.weight.data = self.weight.data * self.mask.data

    def prune_filters(self, prune_index, resample, reinit=False):
        print("fc, prune {} filters".format(len(prune_index)))
        weight_dev = self.weight.device
        mask_dev = self.mask.device
        weight_arr = self.weight.data.cpu().numpy()
        new_mask = self.mask.data.cpu().numpy()
        new_mask[prune_index, :] = 0
        self.weight.data = torch.from_numpy(weight_arr * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)
    
    def mask_on_input_channels(self, prune_index, resample, reinit=False):
        print("fc, mask {} input channels".format(len(prune_index)))
        weight_dev = self.weight.device
        mask_dev = self.mask.device
        weight_arr = self.weight.data.cpu().numpy()
        new_mask = self.mask.data.cpu().numpy()
        new_mask[:, prune_index] = 0
        self.weight.data = torch.from_numpy(weight_arr * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)


class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(MaskedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.groups = groups

        self.weight_prune = Parameter(torch.Tensor(self.out_channels, self.in_channels // self.groups, *self.kernel_size))
        self.mask = Parameter(torch.ones([self.out_channels, self.in_channels // self.groups, *self.kernel_size]), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels=' + str(self.in_channels) \
            + ', out_channels=' + str(self.out_channels) \
            + ', kernel_size=' + str(self.kernel_size) \
            + ', bias=' + str(self.bias is not None) + ')'

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight_prune, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_prune)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.conv2d(input, self.weight_prune, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def prune(self, threshold, resample, reinit=False):
        print("conv, prune weights, percentile_value: {}".format(threshold))
        weight_dev = self.weight_prune.device
        mask_dev = self.mask.device
        
        # Convert Tensors to numpy and calculate
        tensor = self.weight_prune.data.cpu().numpy()
        mask = self.mask.data.cpu().numpy()
        new_mask = np.where(abs(tensor) < threshold, 0, mask)
        
        # Apply new weight and mask
        self.weight_prune.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)

        if reinit:
            nn.init.xavier_uniform_(self.weight_prune)
            self.weight_prune.data = self.weight_prune.data * self.mask.data

    def prune_filters(self, prune_index, resample, reinit=False):
        print("conv, prune {} filters".format(len(prune_index)))
        weight_dev = self.weight_prune.device
        mask_dev = self.mask.device
        weight_arr = self.weight_prune.data.cpu().numpy()
        new_mask = self.mask.data.cpu().numpy()
        new_mask[prune_index, :, :, :] = 0
        self.weight_prune.data = torch.from_numpy(weight_arr * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)

    def mask_on_input_channels(self, prune_index, resample, reinit=False):
        print("mask {} input channels".format(len(prune_index)))
        weight_dev = self.weight_prune.device
        mask_dev = self.mask.device
        weight_arr = self.weight_prune.data.cpu().numpy()
        new_mask = self.mask.data.cpu().numpy()
        new_mask[:, prune_index, :, :] = 0
        self.weight_prune.data = torch.from_numpy(weight_arr * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)


class CustomizedRelu(nn.Module):
    def __init__(self, num_channels, batch_size, inplace=False):
        super(CustomizedRelu, self).__init__()
        self.num_channels = num_channels
        self.inplace = inplace

    def forward(self, input):
        input_channels = list(input.size())[1]
        assert self.num_channels == input_channels
        self.output = F.relu(input, inplace=self.inplace)
        return self.output


    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'num_channels=' + str(self.num_channels) + ')'


class CustomizedRelu6(nn.Module):
    def __init__(self, num_channels, batch_size, inplace=False):
        super(CustomizedRelu6, self).__init__()
        self.num_channels = num_channels
        self.inplace = inplace

    def forward(self, input):
        input_channels = list(input.size())[1]
        assert self.num_channels == input_channels
        self.output = F.relu6(input, inplace=self.inplace)
        return self.output


    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'num_channels=' + str(self.num_channels) + ')'


class MaskedConv2d_MobileNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(MaskedConv2d_MobileNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.groups = groups

        self.weight_prune = Parameter(torch.Tensor(self.out_channels, self.in_channels // self.groups, *self.kernel_size))
        self.mask = Parameter(torch.ones([self.out_channels, self.in_channels // self.groups, *self.kernel_size]), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels=' + str(self.in_channels) \
            + ', out_channels=' + str(self.out_channels) \
            + ', kernel_size=' + str(self.kernel_size) \
            + ', bias=' + str(self.bias is not None) + ')'

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight_prune, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_prune)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.conv2d(input, self.weight_prune, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def prune(self, threshold, resample, reinit=False):
        print("conv, prune weights, percentile_value: {}".format(threshold))
        weight_dev = self.weight_prune.device
        mask_dev = self.mask.device
        # Convert Tensors to numpy and calculate
        tensor = self.weight_prune.data.cpu().numpy()
        mask = self.mask.data.cpu().numpy()
        new_mask = np.where(abs(tensor) < threshold, 0, mask)
        # Apply new weight and mask
        self.weight_prune.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)

        if reinit:
            nn.init.xavier_uniform_(self.weight_prune)
            self.weight_prune.data = self.weight_prune.data * self.mask.data

    def prune_filters(self, prune_index, resample, reinit=False):
        print("conv, prune {} filters".format(len(prune_index)))
        weight_dev = self.weight_prune.device
        mask_dev = self.mask.device
        weight_arr = self.weight_prune.data.cpu().numpy()
        new_mask = self.mask.data.cpu().numpy()
        new_mask[prune_index, :, :, :] = 0
        self.weight_prune.data = torch.from_numpy(weight_arr * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)

    def mask_on_input_channels(self, prune_index, resample, reinit=False):
        print("mask {} input channels".format(len(prune_index)))
        weight_dev = self.weight_prune.device
        mask_dev = self.mask.device
        weight_arr = self.weight_prune.data.cpu().numpy()
        new_mask = self.mask.data.cpu().numpy()
        
        if new_mask.shape[2] == 3:
            new_mask[prune_index, :, :, :] = 0
        elif new_mask.shape[2] == 1:
            prune_index = [e for e in prune_index if e < new_mask.shape[2]] # for shufflenet
            new_mask[:, prune_index, :, :] = 0
        self.weight_prune.data = torch.from_numpy(weight_arr * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)

def _ntuple(n):
    def parse(x):
        if isinstance(x, tuple):
            return x
        return tuple(repeat(x, n))
    return parse

_pair = _ntuple(2)