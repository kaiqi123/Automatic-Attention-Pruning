import time
import torch
import numpy as np
import sys
import os
import json
import logging

def prune(
        curr_round,
        prune_type,
        model,
        resample=False,
        reinit=False,
        args="",
        conv_threshold=-1,
        power_value=-1,
):
    assert args != ""
    assert curr_round != 0
    if "AAP" in prune_type:
        metric_list = model.prune_filters_by_AAP(
            resample=resample, 
            reinit=reinit,
            SAVE_NAME=args.workspace.split("/")[-1], 
            pruning_round=curr_round,
            args=args,
            conv_threshold=conv_threshold,
            power_value=power_value,
            )
    else:
        raise EOFError("Cannot find prune type!")
    return metric_list




def check_optimizer_state(optimizer_state_dict):
    print("=> check_optimizer_state")
    for name in optimizer_state_dict:
        content = optimizer_state_dict[name]
        if isinstance(content, dict):
            print(name, type(content))
        else:
            print(name)
            for k, v in content[0].items():
                if 'params' not in k:
                    print(k, v)
                else:
                    print(k, len(v))
            print("-----")
            for k, v in content[1].items():
                if 'params' not in k:
                    print(k, v)
                else:
                    print(k, len(v))


def check_weights(model):
    for name, p in model.named_parameters():
        if "mask" in name or "relu_para" in name:
            continue
        if "conv" in name:
            print(name, p.shape, p[0][0][0][0])
        if "fc3.weight" in name:
            print(name, p.shape, p[0][0])
        if "fc3.bias" in name:
            print(name, p.shape, p[0])



def load_weights(gpu, rewind_epoch=-1, rewind_filename=None):
    assert rewind_filename != None
    if os.path.isfile(rewind_filename):
        print("=> loading rewinding checkpoint from: {}".format(rewind_filename))
        rewind_checkpoint = torch.load(rewind_filename, map_location=lambda storage, loc: storage.cuda(gpu))
        epoch = rewind_checkpoint["epoch"]
        best_prec1 = rewind_checkpoint["best_prec1"]
        optimizer_state_dict = rewind_checkpoint["optimizer"]
        model_state = rewind_checkpoint["state_dict"]
        assert epoch == rewind_epoch
        print("=> Rewinded, epoch: {}, best_prec1: {}, lr: {}".
              format(epoch, best_prec1, optimizer_state_dict['param_groups'][0]['lr']))
        return optimizer_state_dict, model_state
    else:
        raise EOFError("=> no checkpoint found at '{}'".format(rewind_filename))


def rewind_weights(model, state_dict):
    for name, p in model.named_parameters():
        if "mask" in name or "relu_para" in name:
            continue

        if "conv" in name or "fc" in name:
            if "weight" in name or "weight_prune" in name:
                m = name.replace(f'.{name.split(".")[-1]}', '')
                p.data = model.state_dict()[f"{m}.mask"] * state_dict[name]
            elif "bias" in name:
                p.data = state_dict[name]
            else:
                raise EOFError("name is not correct!")
        else:
            p.data = state_dict[name]

        
def print_nonzeros(model, curr_round, args):
    print("====================================> Parameters <====================================")
    nonzero = total = 0
    fc_total = 0; fc_alive = 0
    conv_total = 0; conv_alive = 0
    filters_dict = {}
    for name, p in model.named_parameters():
        if 'mask' in name or "relu_para" in name or "downsample" in name or "bn" in name or "shortcut" in name:
            continue 
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | '
            f'total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')

        if "fc" in name:
            fc_alive += nz_count
            fc_total += total_params
        if "conv" in name:
            conv_alive += nz_count
            conv_total += total_params
            filters_dict[name] = int(nz_count/np.prod(tensor.shape[1:]))

    pruned = 100 * (total - nonzero) / total
    remained = 100 - pruned
    compress_ratio = total / nonzero
    remained_conv = conv_alive*100/conv_total
    remained_fc = round(fc_alive*100/fc_total, 4)
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  '
          f'({pruned:6.2f}% pruned, {remained:6.2f}% remained; conv: {100-remained_conv:6.2f}% pruned, {remained_conv:6.2f}% remained)')
    
    return remained, remained_fc, remained_conv, compress_ratio


def calculate_flops(model, data_path, args):
    print("====================================> Flops <====================================")
    flops_total_alive = 0
    flops_total_origin = 0
    conv_flops_total_alive = 0
    conv_flops_total_origin = 0
    for name, p in model.named_parameters():

        if "fc3.weight" in name or "weight_prune" in name:
            tensor = p.data.cpu().numpy()
            nz_count = np.count_nonzero(tensor)
            total_params = np.prod(tensor.shape)
            
            if "/tiny-imagenet-200" == data_path:
                if 'vgg' in args.arch:
                    image_size = calculate_image_size_tinyImagenet_vgg(name)
                elif 'resnet101' == args.arch:
                    image_size = calculate_image_size_resnet_bottlenet(name, image_size=64)
                else:
                    raise EOFError(f"On tiny-imagenet-200, {args.arch} is not implemented!") 
            elif "cifar10" in data_path:
                if args.arch in ['resnet18Cifar', 'resnet50Cifar', 'resnet56Cifar']:
                    image_size = calculate_image_size_cifar10_resnet_basicblock(name)
                elif 'resnet101' == args.arch:
                    image_size = calculate_image_size_resnet_bottlenet(name, image_size=32)
                elif 'vgg' in args.arch:
                    image_size = calculate_image_size_cifar10_vgg(name)
                elif 'mobilenetV2' == args.arch:
                    image_size = calculate_image_size_cifar10_mobilenet(name)
                elif 'shufflenetV2' == args.arch:
                    image_size = calculate_image_size_cifar10_shufflenet(name)
                else:
                    raise EOFError(f"On cifar10, {args.arch} is not implemented!")
            else:
                raise EOFError(f"{data_path} is not implemented!")
            
            flops_alive_per_layer = nz_count * 2 * image_size * image_size
            flops_origin_per_layer = total_params * 2 * image_size * image_size
            flops_total_alive += flops_alive_per_layer
            flops_total_origin += flops_origin_per_layer

            if "conv" in name:
                conv_flops_total_alive += flops_alive_per_layer
                conv_flops_total_origin += flops_origin_per_layer

            print(f'{name:20} | alive_flops = {flops_alive_per_layer:7} / {flops_origin_per_layer:7} ({100 * flops_alive_per_layer / flops_origin_per_layer:6.2f}%) | '
                    f'total_pruned = {flops_origin_per_layer - flops_alive_per_layer :7} | shape = {tensor.shape}|'
                    f'image_size={image_size}')
    
    pruned = 100 * (flops_total_origin - flops_total_alive) / flops_total_origin
    flops_remain = 100 - pruned
    flops_speedup = flops_total_origin/flops_total_alive
    conv_pruned = 100 * (conv_flops_total_origin - conv_flops_total_alive) / conv_flops_total_origin
    print(f"===> flops: alive: {flops_total_alive}, pruned: {flops_total_origin-flops_total_alive}, total: {flops_total_origin}, "
          f"Speedup: {flops_speedup:10.2f}x, "
          f'({pruned:6.2f}% pruned, {flops_remain:6.2f}% remained; conv: {conv_pruned:6.2f}% pruned, {100 - conv_pruned:6.2f}% remained)')
    
    return flops_speedup, flops_remain


def calculate_image_size_resnet_bottlenet(name, image_size=64):
    # image_size is 64 for tinyImagenet, 32 for cifar10
    if name == "conv1.weight_prune":
        size = image_size/2

    elif "layer1" in name:
        size = image_size/4
    elif name == "layer2.0.conv1.weight_prune":
        size = image_size/4

    elif "layer2" in name:
        size = image_size/8
    elif name == "layer3.0.conv1.weight_prune":
        size = image_size/8

    elif "layer3" in name:
        size = image_size/16
    elif name == "layer4.0.conv1.weight_prune":
        size = image_size/16

    elif "layer4" in name:
        size = image_size/32
    
    elif "fc" in name:
        size = 1
    
    else:
        print("name is wrong!")
    return size

def calculate_image_size_tinyImagenet_vgg(name):
    if "layer1" in name:
        size = 64

    elif "layer2" in name:
        size = 32

    elif "layer3" in name:
        size = 16

    elif "layer4" in name:
        size = 8

    elif "layer5" in name:
        size = 4

    elif "fc" in name:
        size = 1
    
    else:
        print("name is wrong!")
    return size


def calculate_image_size_cifar10_resnet_basicblock(name):
    if name == "conv1.weight_prune":
        size = 32

    elif "layer1" in name:
        size = 32
    elif name == "layer2.0.conv1.weight_prune" or name == "layer2.0.conv2.weight_prune":
        size = 32

    elif "layer2" in name:
        size = 16
    elif name == "layer3.0.conv1.weight_prune" or name == "layer3.0.conv2.weight_prune":
        size = 16

    elif "layer3" in name:
        size = 8
    elif name == "layer4.0.conv1.weight_prune" or name == "layer4.0.conv2.weight_prune":
        size = 8

    elif "layer4" in name:
        size = 4

    elif "fc" in name:
        size = 1
    
    else:
        print("name is wrong!")
    return size


def calculate_image_size_cifar10_vgg(name):
    if "layer1" in name:
        size = 32

    elif "layer2" in name:
        size = 16

    elif "layer3" in name:
        size = 8

    elif "layer4" in name:
        size = 4

    elif "layer5" in name:
        size = 4

    elif "fc" in name:
        size = 1
    
    else:
        print("name is wrong!")
    return size

def calculate_image_size_cifar10_mobilenet(name):
    layer_size32 = ['0','1','2','3','4','5']
    layer_size16 = ['7','8','9','10','11','12']
    layer_size8 = ['14','15','16']
    block_num = name.split(".")[1]
    if "conv0.weight_prune" in name or block_num in layer_size32 or "layer1.6.conv1" in name:
        size = 32

    elif "layer1.6.conv2" in name or "layer1.6.conv3" in name or block_num in layer_size16 or "layer1.13.conv1" in name:
        size = 16

    elif "layer1.13.conv2" in name or "layer1.13.conv3" in name or block_num in layer_size8 or "conv17.weight_prune" in name:
        size = 8

    elif "fc" in name:
        size = 1
    
    else:
        print("name is wrong!")
    return size

def calculate_image_size_cifar10_shufflenet(name):
    if "conv0.weight_prune" == name:
        size = 32
    elif "layer1" in name:
        size = 16
    elif "layer2" in name:
        size = 8
    elif "layer3" in name or "conv56.weight_prune" == name:
        size = 4
    elif "fc" in name:
        size = 1
    else:
        print("name is wrong!")

    if "layer1.0.conv3.weight_prune" == name:
        size = 32
    if "layer2.0.conv3.weight_prune" == name:
        size = 16
    if "layer3.0.conv3.weight_prune" == name:
        size = 8
    
    return size



def load_weights_from_target_round(args, target_round):
    rewind_filename = f"{args.workspace}/pruningRound_{target_round}_checkpoint.pth.tar"
    if os.path.isfile(rewind_filename):
        print("=> loading rewinding checkpoint from: {}".format(rewind_filename))
        rewind_checkpoint = torch.load(rewind_filename, map_location=lambda storage, loc: storage.cuda(args.gpu))
        epoch = rewind_checkpoint["epoch"]
        best_prec1 = rewind_checkpoint["best_prec1"]
        optimizer_state_dict = rewind_checkpoint["optimizer"]
        loaded_model_state = rewind_checkpoint["state_dict"]
        print("=> Rewinded, epoch: {}, best_prec1: {}, lr: {}".format(epoch, best_prec1, optimizer_state_dict['param_groups'][0]['lr']))
    else:
        raise EOFError("=> no checkpoint found at '{}'".format(rewind_filename))
    return loaded_model_state 

def apdaptive_pruning_given_target(
    args, 
    acc_list,
    para_remained_per_list,
    flops_remained_list,
    conv_threshold_list,
    lambda_list,
    file_thresholds_content,
    curr_round,
    model_and_loss,
    track_rounds_list,
    count_reset_time,
    target,
    curr_reduction,
    one_condition=True, # difference among one and two conditions
    target2=None, # difference among one and two conditions
    curr_reduction2=None, # difference among one and two conditions
    ):

    logging.basicConfig(filename=args.workspace + '/adaptive_threshold_given_target.log', level=logging.DEBUG)
    logging.info(file_thresholds_content)

    assert len(para_remained_per_list) == curr_round+1
    assert len(conv_threshold_list) == curr_round+1


    if curr_reduction <= target:
        if one_condition:
            continue_flag = True
        else:
            assert target2 is not None and curr_reduction2 is not None
            if curr_reduction2 <= target2:
                continue_flag = True
        
        if continue_flag:
            lambda_value = lambda_list[-1]
            conv_threshold = conv_threshold_list[-1] + lambda_value
            track_rounds_list.append(curr_round)
            count_reset_time = 0
            logging.info(f"curr_reduction <= target; curr_reduction={curr_reduction}")
        
    else:
        if not track_rounds_list:
            print("track_rounds_list is empty, exit.")
            sys.exit()
        else:
            index_back = track_rounds_list[-1]

            # reset model
            loaded_model_state = load_weights_from_target_round(args, target_round=index_back)
            model_and_loss.load_model_state(loaded_model_state)
            remained_per_reset2, _, _, _ = print_nonzeros(model_and_loss.get_model(), curr_round, args)
            
            # calculate lambda and threshold
            count_reset_time = count_reset_time + 1
            if count_reset_time == 3:
                lambda_value = lambda_list[-1]/2.0
                conv_threshold = conv_threshold_list[index_back]
                pop_index = track_rounds_list.pop()
                count_reset_time = 0
                assert pop_index == index_back
            else:
                assert count_reset_time < 3
                lambda_value = lambda_list[index_back]/(2**count_reset_time)
                conv_threshold = conv_threshold_list[index_back] + lambda_value
                
            logging.info(f"curr_reduction > target; reset to round: {index_back}; reset times: {count_reset_time}; after resetting, remained_per={remained_per_reset2}")

    logging.info(f"conv_threshold={conv_threshold}, lambda_value={lambda_value}, track_rounds_list: {track_rounds_list}\n")
    return lambda_value, conv_threshold, model_and_loss, track_rounds_list, count_reset_time 