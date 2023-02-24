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
    
    # save_content=f"layer_name={_name}\torg_filters_num={org_filters_num}\t\
    #     remain_num_filters={org_filters_num-len(prune_index)}\tprune_index={prune_index}\tp={power_value}\n"
    # with open(save_dir+f"/round_{pruning_round}.txt", "a") as f:
    #     f.write(save_content)
    save_content={"layer_name": _name, "org_filters_num": org_filters_num, \
        "remain_num_filters": org_filters_num-len(prune_index), "prune_index": prune_index.tolist(), "p": power_value}
    with open(save_dir+f"/round_{pruning_round}.json", "a") as f:
        json.dump(save_content, f)
        f.write("\n")


class PruningModule(Module):
    def prune_by_percentile(self, resample=False, reinit=False, model_name="", **kwargs):
        """
        Note:
             The pruning percentile is based on all layer's parameters concatenated
        Args:
            q (float): percentile in float
            **kwargs: may contain `cuda`
        """
        # Calculate percentile value
        for name, p in self.named_parameters():
            # We do not prune bias, mask, bn, shotchut term
            if 'bias' in name or 'mask' in name or "bn" in name or "downsample" in name or "relu_para" in name:
                continue

            print("\nprune_by_percentile, layer name: {}".format(name))
            tensor = p.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
            # print("111", type(tensor), tensor.shape, alive.shape)

            if "fc" in name:
                new_q = 0
            elif "conv" in name:
                new_q = 20
            else:
                raise ValueError(name)
            percentile_value = np.percentile(abs(alive), new_q)
            print("new_q: {}".format(new_q))

            for _name, module in self.named_modules():
                if name.replace(f'.{name.split(".")[-1]}','') == _name:
                    print(_name, module, new_q)
                    module.prune(threshold=percentile_value, resample=resample, reinit=reinit)

    def prune_next_conv(self, curr_name, prune_index, resample, reinit):
        # build next_conv_dict
        next_conv_dict = {}
        layer_name_list = [layer_name for layer_name, _ in self.named_modules() if "conv" in layer_name and "relu" not in layer_name]
        # assert len(layer_name_list) == 49 or len(layer_name_list) == 19 #imagenet_rn50: 49; cifar10_rn18: 19
        for i in range(0, len(layer_name_list)-1): # except layer4.2.conv3
            next_conv_dict[layer_name_list[i]] = layer_name_list[i+1]
        # for k, v in next_conv_dict.items():
        #     print(k,v)

        # find the next conv module, and put mask on input channel
        for _name, _module in self.named_modules():
            # print(_name, _module) 
            if curr_name in next_conv_dict.keys() and next_conv_dict[curr_name] == _name:
                print(f"curr_name: {curr_name}, next layer name (_name): {_name}")        
                _module.mask_on_input_channels(prune_index=prune_index, resample=resample, reinit=reinit)

    # def prune_filters_by_activations_threshold(self, conv_threshold, resample=False, reinit=False, SAVE_NAME="", pruning_round=-1,**kwargs):
    def prune_filters_by_activations_threshold(
        self, 
        resample=False, 
        reinit=False, 
        SAVE_NAME="", 
        pruning_round=-1, 
        conv_remained_per_list=None, 
        conv_threshold_list=None,
        args="",
        conv_threshold_dict={}, 
        **kwargs,
        ):
        
        std_list = []
        save_content_threshold=f"pruning_round = {pruning_round}\t"
        for name, p in self.named_parameters():
            # We do not prune bias term, and fc3
            if 'bias' in name or 'mask' in name or "fc3" in name or "bn" in name or "downsample" in name or "relu_para" in name or "shortcut" in name:
                continue

            weights_arr = p.data.cpu().numpy() # conv: (out_channel, in_channel, kernal_size, kernal_size)
            print("\nprune_filters_by_activations_threshold, layer name: {}, shape: {}".format(name, weights_arr.shape))
            
            assert  "conv" in name
            # calculate the abs sum of each filter, filers_sum is 1-d array
            if "conv" in name:
                weights_list = list(abs(weights_arr))
                filers_sum = np.array([np.sum(e) for e in weights_list]) # dimension: (out_channel, )
                print(f"filers_sum.shape: {filers_sum.shape}")
            else:
                raise ValueError(name)

            # find its relu, and the index of filters to prune
            for module_name, module in self.named_modules():
                if module_name == name.replace(f'.{name.split(".")[-1]}','') + "_relu":
                    print(module_name, module)
                    # dimension: fc_relu (bs, out_channel), conv_relu (bs, out_channel, image_size, image_size)
                    relu_arr = module.output.data.cpu().numpy() 
                    print("relu_arr.shape: {}".format(relu_arr.shape))
                    assert relu_arr.shape[1] == weights_arr.shape[0]

                    # relu_arr: (bs, out_channel, image_size, image_size)
                    relu_arr_mean_3d = np.mean(relu_arr, axis=0) # (out_channel, image_size, image_size)
                    relu_arr_mean_alive_3d = relu_arr_mean_3d[np.nonzero(filers_sum)] # (alive_channel_num, image_size, image_size)
                    std_relu = np.std(relu_arr_mean_alive_3d, dtype=np.float64)
                    std_list.append(std_relu)

                    # assert len(conv_remained_per_list) == pruning_round
                    # assert len(conv_threshold_list) == pruning_round
                    # if pruning_round >= 3 and conv_remained_per_list[-2] - conv_remained_per_list[-1] < 1:
                    #     conv_threshold = std_relu 
                    # else:
                    #     conv_threshold = conv_threshold_list[-1]
                    conv_threshold = std_relu
                    save_content_threshold = save_content_threshold + f"{name}={round(float(conv_threshold), 4)}\t"
                    print(f"np.nonzero(filers_sum): {np.nonzero(filers_sum)}")
                    print(f"relu_arr_mean_3d.shape: {relu_arr_mean_3d.shape}")
                    print(f"relu_arr_mean_alive_3d.shape: {relu_arr_mean_alive_3d.shape}")
                    print(f"conv_threshold: {conv_threshold}")

                    # filter_relu: (image_size, image_size), relu_arr_mean: (out_channel, )
                    relu_arr_mean = np.array([np.mean(filter_relu) for filter_relu in list(relu_arr_mean_3d)]) 
                    prune_index = np.where(relu_arr_mean <= conv_threshold)[0]
                    print(f"relu_arr_mean.shape: {relu_arr_mean.shape}, prune_index: {prune_index}")

            # find the fc or conv module, and prune filters
            for _name, _module in self.named_modules():
                if name.replace(f'.{name.split(".")[-1]}','') == _name:
                    print(_name, _module, type(_module), type(_name))
                    _module.prune_filters(prune_index=prune_index, resample=resample, reinit=reinit)

                    # calculate and save remaining filters to test inference time
                    print("=====>Save number of remaining filters for inference time")
                    assert pruning_round != -1
                    save_models_for_inference_time(_name, _module, prune_index, pruning_round, SAVE_NAME, power_value=-1)
            # sys.exit()
        
        # save threhold of each layer
        with open(args.workspace+"/threshold_per_layer.txt", "a") as f:
            f.write(save_content_threshold+"\n")
        return conv_threshold, std_list, conv_threshold_dict

    def prune_filters_by_activations_threshold_uniform(
        self, 
        resample=False, 
        reinit=False, 
        SAVE_NAME="", 
        pruning_round=-1, 
        args="",
        conv_threshold=-1,
        power_value=1,
        # conv_remained_per_list=None, 
        # conv_threshold_list=None,
        # acc_list=None,
        # conv_threshold_dict={}, 
        **kwargs,
        ):
        
        # calculate total_alive_relu, then calculate std and mean of total_alive_relu
        print("1. Calculate total_alive_relu, then calculate std and mean of total_alive_relu")
        total_alive_relu = []
        # total_alive_relu2 = []
        # total_alive_relu_std = []
        for name, p in self.named_parameters():
            # We do not prune bias term, and fc3
            if 'bias' in name or 'mask' in name or "fc3" in name or "bn" in name or "downsample" in name or "relu_para" in name or "shortcut" in name:
                continue
            
            weights_arr = p.data.cpu().numpy() # conv: (out_channel, in_channel, kernal_size, kernal_size)
            # print("\nprune_filters_by_activations_threshold_uniform, layer name: {}, shape: {}".format(name, weights_arr.shape))
            
            assert  "conv" in name
            # calculate the abs sum of each filter, filers_sum is 1-d array
            if "conv" in name:
                weights_list = list(abs(weights_arr))
                filers_sum = np.array([np.sum(e) for e in weights_list]) # dimension: (out_channel, )
                # print(f"filers_sum.shape: {filers_sum.shape}")
            else:
                raise ValueError(name)

            # find its alive relu_arr
            for module_name, module in self.named_modules():
                if module_name == name.replace(f'.{name.split(".")[-1]}','') + "_relu":
                    # print(module_name, module)
                    # dimension: fc_relu (bs, out_channel), conv_relu (bs, out_channel, image_size, image_size)
                    relu_arr = module.output.data.cpu().numpy() 
                    # print("relu_arr.shape: {}".format(relu_arr.shape))
                    assert relu_arr.shape[1] == weights_arr.shape[0]
           
                    # method 1 for calculating std
                    # relu_arr: (bs, out_channel, image_size, image_size)
                    # relu_arr_mean_3d = np.mean(relu_arr, axis=0) # (out_channel, image_size, image_size)
                    # relu_arr_mean_alive_3d = relu_arr_mean_3d[np.nonzero(filers_sum)] # (alive_channel_num, image_size, image_size)
                    # relu_arr_mean_alive_3d_list = list(relu_arr_mean_alive_3d.flatten())
                    # assert len(relu_arr_mean_alive_3d_list) == relu_arr_mean_alive_3d.shape[0] * relu_arr_mean_alive_3d.shape[1] * relu_arr_mean_alive_3d.shape[2]
                    # total_alive_relu1 = total_alive_relu1 + relu_arr_mean_alive_3d_list
                    # print(f"length of relu_arr_mean_alive_3d_list: {len(relu_arr_mean_alive_3d_list)}")
                    # print(f"length of total_alive_relu1: {len(total_alive_relu1)}")
                    # print(f"relu_arr_mean_3d.shape: {relu_arr_mean_3d.shape}")
                    # print(f"relu_arr_mean_alive_3d.shape: {relu_arr_mean_alive_3d.shape}")

                    # method 2 for calculating std
                    # relu_arr: (bs, out_channel, image_size, image_size)
                    relu_arr_mean_3d = np.mean(relu_arr, axis=0) # (out_channel, image_size, image_size)
                    alive_index = list(np.nonzero(filers_sum)[0])
                    img_arr_org = relu_arr_mean_3d
                    mean_list = [] # the mean value of the activations of each filter
                    for num_channel in range(0, img_arr_org.shape[0]):
                        img_arr = img_arr_org[num_channel] # image for one filter: (image_size, image_size)
                        img_arr_mean = np.mean(img_arr)
                        mean_list.append(img_arr_mean)
                        # print(num_channel, img_arr.shape, img_arr_mean)
                    assert len(mean_list) == img_arr_org.shape[0]
                    alive_mean_list = list(np.array(mean_list)[alive_index])
                    total_alive_relu = total_alive_relu + alive_mean_list
                    
                    # std_total_alive_relu1 = np.std(total_alive_relu1, dtype=np.float64)
                    # std_total_alive_relu2 = np.std(total_alive_relu2, dtype=np.float64)
                    # print(std_total_alive_relu1, std_total_alive_relu2)
                    
                    # relu_arr: (bs, out_channel, image_size, image_size), filter_relu: (image_size, image_size)
                    # relu_arr_mean_3d_list = list(np.mean(relu_arr, axis=0)) # (out_channel, )
                    # relu_std = np.array([np.std(filter_relu, dtype=np.float64) for filter_relu in relu_arr_mean_3d_list]) # (out_channel, )
                    # relu_std_alive = list(relu_std[np.nonzero(filers_sum)]) # list length: alive_out_channel
                    # total_alive_relu_std = total_alive_relu_std + relu_std_alive
                    # print(f"length of relu_arr_mean_3d_list: {len(relu_arr_mean_3d_list)}")
                    # print(f"length of relu_std_alive: {len(relu_std_alive)}")
                    # print(f"length of total_alive_relu_std: {len(total_alive_relu_std)}")
        
        # calculate std according to total_alive_relu
        std_total_alive_relu = np.std(total_alive_relu, dtype=np.float64)
        mean_total_alive_relu = np.mean(total_alive_relu)
        # conv_threshold = std_total_alive_relu * 0.25
        
        # assert len(conv_remained_per_list) == pruning_round
        # assert len(conv_threshold_list) == pruning_round
        # if pruning_round >= 3 and conv_remained_per_list[-2] - conv_remained_per_list[-1] < 1:
        #     conv_threshold = conv_threshold_list[-1]+0.01 # 0.2 in paper
        # else:
        #     conv_threshold = conv_threshold_list[-1]
        # if conv_threshold>0.18:
        #     conv_threshold=0.18

        # increase0.01Max0.1
        # if pruning_round >= 1:
        #     conv_threshold = conv_threshold_list[-1] + 0.02
        # if conv_threshold > 0.3:
        #     conv_threshold = 0.3

        # conv_threshold = 0.4
        
        # acc_divider = min(acc_list[0], acc_list[-1])
        # gl = 100.0 * (acc_list[0]/acc_divider - 1.0)
        # conv_threshold = 0.5 * (1.0 - 1.0 / (1.0 + gl)) 
        print("=====================>conv_threshold: {}".format(conv_threshold))

        # new_q = 5
        # total_alive_relu_std_percentile_value = np.percentile(total_alive_relu_std, new_q)
        # conv_threshold = total_alive_relu_std_percentile_value
        # print(f"length of total_alive_relu_std: {len(total_alive_relu_std)}")
        # print(f"pruning rate is: {new_q}, total_alive_relu_std_percentile_value: {total_alive_relu_std_percentile_value}")
        # print("=====================>conv_threshold: {}".format(conv_threshold))
        
        # Calculate weights of each layer according to its number of parameters or flops
        print("2. Calculate weights of each layer according to its number of parameters")
        para_total_nz = 0.0
        flops_total_nz = 0.0
        threshold_weights_dict_para = {}
        threshold_weights_dict_flops = {}
        for name, module in self.named_modules():
            if "relu" in name and "conv" in name:
                # print("module name: {}, shape: {}".format(name, relu_arr.shape))
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
                            if "/imagenet" == args.data:
                                if 'resnet' in args.arch:
                                    image_size = pruning_utils.calculate_image_size_imagenet_resnet(parameter_name)
                                else:
                                    raise EOFError("On imagenet, args.arch is wrong !")
                            elif "/tiny-imagenet-200" == args.data:
                                if 'vgg' in args.arch:
                                    image_size = pruning_utils.calculate_image_size_tinyImagenet_vgg(parameter_name)
                                elif 'resnet101' == args.arch or 'resnet152' == args.arch:
                                    image_size = pruning_utils.calculate_image_size_bottlenet_resnet(parameter_name, image_size=64)
                                elif 'resnet50' == args.arch:
                                    image_size = pruning_utils.calculate_image_size_imagenet_resnet(parameter_name, image_size=64)
                                else:
                                    raise EOFError(f"On tiny-imagenet-200, {args.arch} is wrong !")  
                            elif "cifar10" in args.data:
                                if args.arch in ['resnet18Cifar', 'resnet50Cifar', 'resnet56Cifar']:
                                    image_size = pruning_utils.calculate_image_size_cifar10_resnet(parameter_name)
                                elif 'resnet101' == args.arch or 'resnet152' == args.arch:
                                    image_size = pruning_utils.calculate_image_size_bottlenet_resnet(parameter_name, image_size=32)
                                elif 'vgg' in args.arch:
                                    image_size = pruning_utils.calculate_image_size_cifar10_vgg(parameter_name)
                                elif 'mobilenetV2' == args.arch:
                                    image_size = pruning_utils.calculate_image_size_cifar10_mobilenet(parameter_name)
                                elif 'shufflenetV2' == args.arch:
                                    image_size = pruning_utils.calculate_image_size_cifar10_shufflenet(parameter_name)
                                else:
                                    raise EOFError("On cifar10, args.arch is wrong !")
                            else:
                                raise EOFError("data_path is wrong !")
                            flops_curr_nz = para_curr_nz * 2 * image_size * image_size
                            flops_total_nz = flops_total_nz + flops_curr_nz
                            threshold_weights_dict_flops[name] = flops_curr_nz
                            # print(parameter_name, p.shape, nz_count)
        
        for k,v in threshold_weights_dict_para.items():
            threshold_weights_dict_para[k] = v / para_total_nz
        for k,v in threshold_weights_dict_flops.items():
            threshold_weights_dict_flops[k] = v / flops_total_nz
        
        # for k,v in threshold_weights_dict_flops.items():
        #     print(k, v)
        # print(round(sum(threshold_weights_dict_flops.values()), 2))
        # sys.exit()

        assert round(sum(threshold_weights_dict_para.values()), 2) == 1.0
        assert round(sum(threshold_weights_dict_flops.values()), 2) == 1.0

        # prune filters in each layer
        print("3. Prune filters in each layer")
        save_content_threshold=f"pruning_round = {pruning_round}\t"
        for name, module in self.named_modules():
            if "relu" in name and "conv" in name:

                # dimension: fc_relu (bs, out_channel), conv_relu (bs, out_channel, image_size, image_size)
                relu_arr = module.output.data.cpu().numpy()
                print("\nprune_filters_by_activations_threshold_uniform, module name: {}, shape: {}".format(name, relu_arr.shape))

                # relu_arr_mean = np.array([np.mean(filter_relu) for filter_relu in list(np.mean(relu_arr, axis=0))])
                # relu_arr_mean = np.array([np.std(filter_relu, dtype=np.float64) for filter_relu in list(np.mean(relu_arr, axis=0))])
                relu_arr_mean = np.array([np.mean(np.power(filter_relu, power_value)) for filter_relu in list(np.mean(relu_arr, axis=0))])
                
                # prune_index = np.where(relu_arr_mean <= conv_threshold)[0]
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

        # metric_list = [std_total_alive_relu1, std_total_alive_relu2]
        metric_list = [std_total_alive_relu, mean_total_alive_relu]
        # std_list = [np.std(total_alive_relu_std, dtype=np.float64), np.mean(total_alive_relu_std), np.min(total_alive_relu_std), np.max(total_alive_relu_std)]
        return metric_list

    def prune_filters_by_activations_global(self, resample=False, reinit=False, SAVE_NAME="", pruning_round=-1, power_value=-1, pruning_rate=20, **kwargs):
        
        total_relu_arr_mean_alive = []
        count_layer = 0
        for name, p in self.named_parameters():
            
            # We do not prune bias term, and fc3
            if 'bias' in name or 'mask' in name or "fc3" in name or "bn" in name or "downsample" in name or "relu_para" in name or "shortcut" in name:
                continue

            weights_arr = p.data.cpu().numpy() # conv: (out_channel, in_channel, kernal_size, kernal_size)
            print("\n prune_filters_by_activations_global, layer name: {}, shape: {}".format(name, weights_arr.shape))
            
            assert  "conv" in name
            assert  "fc" not in name
            weights_list = list(abs(weights_arr))
            filers_sum = np.array([np.sum(e) for e in weights_list]) # length: out_channel
            # print(len(weights_list), weights_list[0].shape)

            # find its relu, and add each alive relu activation to total_relu_arr_mean_alive
            for module_name, module in self.named_modules():
                if module_name == name.replace(f'.{name.split(".")[-1]}','') + "_relu":
                    print(module_name, module)
                    # relu_arr: fc (bs, out_channel), conv (bs, out_channel, image_size, image_size)
                    relu_arr = module.output.data.cpu().numpy() 
                    print("relu_arr.shape: {}".format(relu_arr.shape))
                    assert relu_arr.shape[1] == weights_arr.shape[0]
                    
                    # if name.startswith("fc"): # relu_arr: (bs, out_channel)
                    #     relu_arr_mean = np.mean(relu_arr, axis=0) # (out_channel, )
                    print("power_value: {}".format(power_value))
                    assert power_value == 1
                    # relu_arr: (bs, out_channel, image_size, image_size); 
                    # filter_relu: (image_size, image_size); relu_arr_mean: (out_channel, )
                    # Note: use np.mean here; use np.sum in prune_filters_by_activations()
                    # relu_arr_mean = np.array([np.sum(filter_relu) for filter_relu in list(np.mean(relu_arr, axis=0))])
                    relu_arr_mean = np.array([np.mean(np.power(filter_relu, power_value)) for filter_relu in list(np.mean(relu_arr, axis=0))])
                    relu_arr_mean_alive = list(relu_arr_mean[np.nonzero(filers_sum)])
                    total_relu_arr_mean_alive = total_relu_arr_mean_alive + relu_arr_mean_alive
                    print(relu_arr_mean.shape, len(relu_arr_mean_alive))
            count_layer = count_layer + 1
        
        # find the smallest alive relu activations
        assert count_layer == 49
        new_q = pruning_rate
        total_relu_mean_percentile_value = np.percentile(total_relu_arr_mean_alive, new_q)
        print(f"length of total_relu_arr_mean_alive: {len(total_relu_arr_mean_alive)}")
        print(f"pruning rate is: {new_q}, total_relu_mean_percentile_value: {total_relu_mean_percentile_value}")
        # sys.exit()

        # prune activations according to total_relu_mean_percentile_value
        for name, module in self.named_modules():
            if "relu" in name and "conv" in name:
                # dimension: fc_relu (bs, out_channel), conv_relu (bs, out_channel, image_size, image_size)
                relu_arr = module.output.data.cpu().numpy()
                print("\n prune_filters_by_activations_global, module name: {}, shape: {}".format(name, relu_arr.shape))

                assert power_value == 1
                relu_arr_mean = np.array([np.mean(np.power(filter_relu, power_value)) for filter_relu in list(np.mean(relu_arr, axis=0))])
                prune_index = [i for i in range(len(relu_arr_mean)) if relu_arr_mean[i] <= total_relu_mean_percentile_value]
                # prune_index = np.where(relu_arr_mean <= total_relu_mean_percentile_value)[0]  # index_zero: list
                print("relu_arr_mean.shape: {}, average of relu_arr_mean: {}".format(relu_arr_mean.shape, np.mean(relu_arr_mean)))

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
                        # print("=====>Save number of remaining filters for inference time")
                        # assert pruning_round != -1
                        # save_models_for_inference_time(_name, _module, prune_index, pruning_round, SAVE_NAME, power_value=-1)
        


    def prune_filters_by_activations(self, resample=False, reinit=False, SAVE_NAME="", pruning_round=-1, power_value=-1, pruning_rate=20, **kwargs):
        for name, p in self.named_parameters():
            # We do not prune bias term, and fc3
            if 'bias' in name or 'mask' in name or "fc3" in name or "bn" in name or "downsample" in name or "relu_para" in name or "shortcut" in name:
                continue

            weights_arr = p.data.cpu().numpy() # fc: (300, 784), conv: (64, 3, 3, 3) (out_channel, in_channel, kernal_size, kernal_size)
            print("\nprune_filters_by_activations, layer name: {}, shape: {}".format(name, weights_arr.shape))
            
            assert  "conv" in name
            # calculate the abs sum of each filter, filers_sum is 1-d array
            # if "fc" in name:
            #     new_q = 0.0
            #     filers_sum = np.sum(abs(weights_arr), axis=1)
            if "conv" in name:
                new_q = pruning_rate
                weights_list = list(abs(weights_arr))
                filers_sum = np.array([np.sum(e) for e in weights_list]) # length: 64, out_channel
                # print(len(weights_list), weights_list[0].shape)
            else:
                raise ValueError(name)
            print("pruning rate is: {}".format(new_q))

            # previous implementation, calcualte the number of filters will be pruned, new + already die
            # alive_num = np.count_nonzero(filers_sum) # if weights are 0, its abs sum is 0
            # prune_num = round(alive_num * new_q) + (weights_arr.shape[0] - alive_num)  # new prune number + die number
            # print(f"alive_num: {alive_num}, die num: {(weights_arr.shape[0] - alive_num)}, "
            #       f"new prune mum: {round(alive_num * new_q)}, prune_num (new prune + die): {prune_num}")

            # find its relu, and the index of filters to prune
            for module_name, module in self.named_modules():
                if module_name == name.replace(f'.{name.split(".")[-1]}','') + "_relu":
                    print(module_name, module)
                    # relu_arr = module.relu_para.data.cpu().numpy() # dimension: fc_relu (bs, out_channel), conv_relu (bs, out_channel, image_size, image_size)
                    relu_arr = module.output.data.cpu().numpy() # dimension: fc_relu (bs, out_channel), conv_relu (bs, out_channel, image_size, image_size)
                    print("relu_arr.shape: {}".format(relu_arr.shape))
                    assert relu_arr.shape[1] == weights_arr.shape[0]
                    
                    # fc: (300, ); conv: relu_list [num_channel, image_size, image_size], filter_relu [image_size, image_size]
                    # original prune by avtivations
                    # relu_arr_mean = np.mean(relu_arr, axis=0) if name.startswith("fc") else \
                        # np.array([np.sum(filter_relu) for filter_relu in list(np.mean(relu_arr, axis=0))])
                    
                    # prune by attention, only apply to conv
                    if name.startswith("fc"):
                        # relu_arr: (bs, out_channel)
                        relu_arr_mean = np.mean(relu_arr, axis=0) # (out_channel, )
                    else:
                        # relu_arr: (bs, out_channel, image_size, image_size)

                        # relu_arr_trans = relu_arr.transpose(1,0,2,3) # (out_channel, bs, image_size, image_size)
                        # img_arr_org = np.mean(relu_arr_trans, axis=1) # (out_channel, image_size, image_size)
                        # relu_arr_mean = [] # list: (out_channel)
                        # for num_channel in range(0, img_arr_org.shape[0]):
                        #     img_arr = img_arr_org[num_channel] # img_arr is image for one filter: (image_size, image_size)
                        #     img_arr_p = np.power(img_arr, power_value)
                        #     relu_arr_mean.append(np.sum(img_arr_p))
                        # print(relu_arr_trans.shape, img_arr_org.shape, len(relu_arr_mean), power_value)

                        # org, prune by activations
                        # assert power_value == -1
                        # relu_arr_mean = np.array([np.sum(filter_relu) for filter_relu in list(np.mean(relu_arr, axis=0))])
                        print("power_value: {}".format(power_value))
                        relu_arr_mean = np.array([np.mean(np.power(filter_relu, power_value)) for filter_relu in list(np.mean(relu_arr, axis=0))])
                        # relu_arr_mean = np.array([np.sum(np.power(filter_relu, power_value)) for filter_relu in list(np.mean(relu_arr, axis=0))])
                        # relu_arr_mean = np.array([np.max(np.power(filter_relu, power_value)) for filter_relu in list(np.mean(relu_arr, axis=0))])

                    relu_arr_mean_alive = relu_arr_mean[np.nonzero(filers_sum)]
                    relu_arr_mean_percentile_value = np.percentile(relu_arr_mean_alive, new_q)
                    prune_index = [i for i in range(len(relu_arr_mean)) if relu_arr_mean[i] <= relu_arr_mean_percentile_value]

                    # previous implementation
                    # relu_arr_mean_percentile_value = np.percentile(relu_arr_mean, prune_num / relu_arr.shape[1] * 100)  # include already died
                    # prune_index = [i for i in range(len(relu_arr_mean)) if relu_arr_mean[i] <= relu_arr_mean_percentile_value]
                    
                    # print(relu_arr.shape)
                    # print(relu_arr_mean.shape, relu_arr_mean)
                    # print(f"relu_arr_mean_percentile_value: {relu_arr_mean_percentile_value}, f"prune_num / relu_arr.shape[1] * 100 : {prune_num / relu_arr.shape[1] * 100}")

            print("====>Begin to find the next conv module, and put mask on input channel")
            self.prune_next_conv(
                curr_name=name.replace(f'.{name.split(".")[-1]}',''), 
                prune_index=prune_index, 
                resample=resample, 
                reinit=reinit)

            print("====>Begin to find ind the fc or conv module, and prune filters (put mask on output channel)")
            for _name, _module in self.named_modules():
                if name.replace(f'.{name.split(".")[-1]}','') == _name:
                    print(_name, _module, type(_module), type(_name))
                    _module.prune_filters(prune_index=prune_index, resample=resample, reinit=reinit)

                    # calculate and save remaining filters to test inference time
                    # print("=====>Save number of remaining filters for inference time")
                    # assert pruning_round != -1
                    # save_models_for_inference_time(_name, _module, prune_index, pruning_round, SAVE_NAME, power_value)
        # sys.exit()

    def prune_filters_by_l1norm(self, resample=False, reinit=False, SAVE_NAME="", pruning_round=-1, pruning_rate=20, **kwargs):
        for name, p in self.named_parameters():
            # We do not prune bias term, and fc3
            if 'bias' in name or 'mask' in name or "fc3" in name or "bn" in name or "downsample" in name or "relu_para" in name or "shortcut" in name:
                continue

            weights_arr = p.data.cpu().numpy() # fc: (300, 784), conv: (64, 3, 3, 3) (out_channel, in_channel, kernal_size, kernal_size)
            print("\nprune_filters_by_l1norm, layer name: {}, shape: {}".format(name, weights_arr.shape))

            # calculate the abs sum of each filter, filers_sum is 1-d array
            # if "fc" in name:
            #     new_q = 0
            #     filers_sum = np.sum(abs(weights_arr), axis=1)  # length: 300
            if "conv" in name:
                new_q = pruning_rate
                weights_list = list(abs(weights_arr))
                filers_sum = np.array([np.sum(e) for e in weights_list]) # length: 64, out_channel
                # print(len(weights_list), weights_list[0].shape)
            else:
                raise ValueError(name)
            print("pruning rate is: {}".format(new_q))

            filers_sum_alive = filers_sum[np.nonzero(filers_sum)]
            filers_sum_percentile_value = np.percentile(filers_sum_alive, new_q)
            prune_index = [i for i in range(len(filers_sum)) if filers_sum[i] <= filers_sum_percentile_value]
            # print(len(filers_sum), filers_sum)
            # print(len(filers_sum_alive), filers_sum_alive)
            # print(filers_sum_percentile_value)
            # print(len(prune_index), prune_index)

            print("====>Begin to find the next conv module, and put mask on input channel")
            self.prune_next_conv(
                curr_name=name.replace(f'.{name.split(".")[-1]}',''), 
                prune_index=prune_index, 
                resample=resample, 
                reinit=reinit)

            print("====>Begin to find ind the fc or conv module, and prune filters (put mask on output channel)")
            for _name, _module in self.named_modules():
                if name.replace(f'.{name.split(".")[-1]}','') == _name:
                    print(_name, _module, new_q)
                    _module.prune_filters(prune_index=prune_index, resample=resample, reinit=reinit)

                    # calculate and save remaining filters to test inference time
                    # print("=====>Save number of remaining filters for inference time")
                    # assert pruning_round != -1
                    # assert save_models_for_inference_time != ""
                    # save_models_for_inference_time(_name, _module, prune_index, pruning_round, save_name_inference_time)
                    # save_models_for_inference_time(_name, _module, prune_index, pruning_round, SAVE_NAME, power_value=-1)
            
    def prune_by_std(self, s=0.25):
        """
        Note that `s` is a quality parameter / sensitivity value according to the paper.
        According to Song Han's previous paper (Learning both Weights and Connections for Efficient Neural Networks),
        'The pruning threshold is chosen as a quality parameter multiplied by the standard deviation of a layer’s weights'
        I tried multiple values and empirically, 0.25 matches the paper's compression rate and number of parameters.
        Note : In the paper, the authors used different sensitivity values for different layers.
        """
        for name, module in self.named_modules():
            if name in ['fc1', 'fc2', 'fc3']:
                threshold = np.std(module.weight.data.cpu().numpy()) * s
                print(f'Pruning with threshold : {threshold} for layer {name}')
                module.prune(threshold)


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
        # print(tensor.shape, mask.shape, new_mask.shape)
        if resample:
            new_mask = np.random.permutation(new_mask)
        # Apply new weight and mask
        self.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)

        if reinit:
            nn.init.xavier_uniform_(self.weight)
            self.weight.data = self.weight.data * self.mask.data

    # fc layer prune filters, not need for resnet
    # def prune_filters(self, prune_index, resample, reinit=False):
    #     print("fc, prune {} filters".format(len(prune_index)))
    #     weight_dev = self.weight.device
    #     mask_dev = self.mask.device
    #     weight_arr = self.weight.data.cpu().numpy()
    #     new_mask = self.mask.data.cpu().numpy()
    #     # print(new_mask.shape, new_mask[prune_index, :])
    #     new_mask[prune_index, :] = 0
    #     # print(new_mask.shape, np.where(new_mask == 0)[0])
    #     self.weight.data = torch.from_numpy(weight_arr * new_mask).to(weight_dev)
    #     self.mask.data = torch.from_numpy(new_mask).to(mask_dev)


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

    # def prune(self, threshold, reinit=False):
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
        # print(tensor.shape, mask.shape)

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
        # self.batch_size = batch_size
        # self.relu_para = Parameter(torch.Tensor(self.batch_size, self.num_channels), requires_grad=False)

    def forward(self, input):
        input_channels = list(input.size())[1]
        assert self.num_channels == input_channels
        self.output = F.relu(input, inplace=self.inplace)

        # self.relu_para.data = self.output[:self.batch_size].to(self.output.device)
        return self.output


    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'num_channels=' + str(self.num_channels) + ')'


class CustomizedRelu6(nn.Module):
    def __init__(self, num_channels, batch_size, inplace=False):
        super(CustomizedRelu6, self).__init__()
        self.num_channels = num_channels
        self.inplace = inplace
        # self.batch_size = batch_size
        # self.relu_para = Parameter(torch.Tensor(self.batch_size, self.num_channels), requires_grad=False)

    def forward(self, input):
        input_channels = list(input.size())[1]
        assert self.num_channels == input_channels
        self.output = F.relu6(input, inplace=self.inplace)

        # self.relu_para.data = self.output[:self.batch_size].to(self.output.device)
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

    # def prune(self, threshold, reinit=False):
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
        # print(tensor.shape, mask.shape)

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
        
        # difference  
        if new_mask.shape[2] == 3:
            new_mask[prune_index, :, :, :] = 0
        elif new_mask.shape[2] == 1:
            # print(new_mask.shape)
            # print(prune_index)
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