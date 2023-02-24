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
        model_name,
        resample=False,
        reinit=False,
        args="",
        conv_threshold=-1,
        power_value=-1,
        # conv_threshold_list=[],
        # conv_remained_per_list=[],
        # acc_list = [],
        # std_list = [],
        # conv_threshold_dict = {},
):
    assert args != ""
    print(f"\n => Prune, defaults for pruning level {curr_round}")
    print("curr_round: {}\nprune_type: {}\nmodel_name: {}: \nresample: {}\nreinit: {}\n".
        format(curr_round, prune_type, model_name, resample, reinit))

    assert curr_round != 0
    if "nonstructure" in prune_type:
        model.prune_by_percentile(resample=resample, reinit=reinit, model_name=model_name)
    elif "ActivThreshold" in prune_type:
        # threshold = set_conv_threshold(conv_remained_per_list, conv_threshold_list, curr_round)
        # threshold = 0.0
        # model.prune_filters_by_activations_threshold(
        #     conv_threshold=threshold, 
        #     resample=resample, 
        #     reinit=reinit,
        #     SAVE_NAME=args.workspace.split("/")[-1], 
        #     pruning_round=curr_round,
        # )
        metric_list = model.prune_filters_by_activations_threshold_uniform(
            resample=resample, 
            reinit=reinit,
            SAVE_NAME=args.workspace.split("/")[-1], 
            pruning_round=curr_round,
            args=args,
            conv_threshold=conv_threshold,
            power_value=power_value,
            # conv_remained_per_list=conv_remained_per_list, 
            # conv_threshold_list=conv_threshold_list,
            # acc_list=acc_list,
            # conv_threshold_dict=conv_threshold_dict,
            )
    elif "Activ" in prune_type:
        model.prune_filters_by_activations(
        # model.prune_filters_by_activations_global(
            resample=resample, 
            reinit=reinit, 
            SAVE_NAME=args.workspace.split("/")[-1], 
            pruning_round=curr_round, 
            power_value=args.power_value,
            pruning_rate=args.pruning_rate,
            )
        metric_list = [1,1]
    elif "L1norm" in prune_type:
        model.prune_filters_by_l1norm(
            resample=resample, 
            reinit=reinit, 
            SAVE_NAME=args.workspace.split("/")[-1], 
            pruning_round=curr_round,
            pruning_rate=args.pruning_rate,
            )
        metric_list = [1,1]
    else:
        raise EOFError("Cannot find prune type!")
    return metric_list


# def set_conv_threshold(conv_remained_per_list, conv_thresholds, curr_round):
#     # print(conv_remained_per_list, conv_thresholds, curr_round)
#     assert len(conv_remained_per_list) == curr_round
#     assert len(conv_thresholds) == curr_round
#     if curr_round >= 3 and conv_remained_per_list[-2] - conv_remained_per_list[-1] < 1:
#         return conv_thresholds[-1]+0.01 # 0.2 in paper
#     else:
#         return conv_thresholds[-1]


def check_optimizer_state(optimizer_state_dict):
    print("=> check_optimizer_state")
    for name in optimizer_state_dict:
        content = optimizer_state_dict[name]
        if isinstance(content, dict):
            # print(name, content.keys())
            print(name, type(content))
            # for k, v in content.items():
            #     print(k, v.keys())
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


# def rewind_weights_and_optimizer(model, gpu, optimizer, rewind_epoch=-1, rewind_filename=None):
#     assert rewind_filename != None
#     if os.path.isfile(rewind_filename):
#         print("=> loading rewinding checkpoint from: {}".format(rewind_filename))
#         rewind_checkpoint = torch.load(rewind_filename, map_location=lambda storage, loc: storage.cuda(gpu))
#         epoch = rewind_checkpoint["epoch"]
#         best_prec1 = rewind_checkpoint["best_prec1"]
#         optimizer_state_dict = rewind_checkpoint["optimizer"]
#         rewind_weights(model, rewind_checkpoint["state_dict"])
#         optimizer.load_state_dict(optimizer_state_dict)
#         assert epoch == rewind_epoch
#         print("=> Rewinded, epoch: {}, best_prec1: {}, lr: {}".
#               format(epoch, best_prec1, optimizer_state_dict['param_groups'][0]['lr']))
#         return optimizer
#     else:
#         raise EOFError("=> no checkpoint found at '{}'".format(rewind_filename))

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
                # print("a", name)
            elif "bias" in name:
                # print("b", name)
                # p.data = model.state_dict()[name]
                p.data = state_dict[name]
            else:
                raise EOFError("name is not correct!")
        else:
            # print("c", name)
            # p.data = model.state_dict()[name]
            p.data = state_dict[name]

# save remaining number of weights for inference time analysis
# save to "./inferenece_time_measure/model_results/{xxxxxx}/"
def save_weightsNum_for_inference_time(pruning_round, save_path, remained_per, remained_conv, filters_dict):
    print("===============>save_weightsNum_for_inference_time<====================")
    assert pruning_round != -1

    # save_content_filter=""
    # for name, num in filters_dict.items():
    #     # print(name[:-13], num) # name[:-13]: remove .weight_prune
    #     save_content_filter = save_content_filter + name[:-13] + f"={num}\t"
    # save_content=f"pruning_round={pruning_round}\tremained_per={remained_per}\tremained_conv={remained_conv}\t"+save_content_filter+"\n"
    # print(save_content)
    # with open(save_path, "a") as f:
    #     f.write(save_content)
 
    with open(save_path, "a") as f:
        json.dump({"pruning_round": pruning_round, "remained_per": remained_per, "remained_conv": remained_conv}, f)
        f.write("\n")
        save_content_filter=""
        for name, num in filters_dict.items():
            # print(name[:-13], num) # name[:-13]: remove .weight_prune
            json.dump({str(name[:-13]): num}, f)
            f.write("\n")
        
def print_nonzeros(model, curr_round, args):
    nonzero = total = 0
    fc_total = 0; fc_alive = 0
    conv_total = 0; conv_alive = 0
    filters_dict = {} # for saving filters to analyze inference time
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
    
    # save remaining number of weights for inference time analysis
    # save to "./inferenece_time_measure/model_results/{xxxxx}/remained_weightsNum_{curr_round}.json"
    # save_dir = "./inferenece_time_measure/model_results/"+args.workspace.split("/")[-1]
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # save_path = save_dir+f"/remained_weightsNum_{curr_round}.json"
    # save_weightsNum_for_inference_time(curr_round, save_path, remained, remained_conv, filters_dict)
    return remained, remained_fc, remained_conv, compress_ratio


def calculate_flops(model, data_path, args):
    print("====================================>Flops<====================================")
    flops_total_alive = 0
    flops_total_origin = 0
    conv_flops_total_alive = 0
    conv_flops_total_origin = 0
    for name, p in model.named_parameters():
        # only calculate fc.weight and conv.weight, ignore bias and others
        if "fc3.weight" in name or "weight_prune" in name:
            tensor = p.data.cpu().numpy()
            nz_count = np.count_nonzero(tensor)
            total_params = np.prod(tensor.shape)
            
            # calculate flops including fc layer
            # print(data_path, args.arch)
            if "/imagenet" == data_path:
                if 'resnet' in args.arch:
                    image_size = calculate_image_size_imagenet_resnet(name)
                else:
                    raise EOFError("On imagenet, args.arch is wrong !")
            elif "/tiny-imagenet-200" == data_path:
                if 'vgg' in args.arch:
                    image_size = calculate_image_size_tinyImagenet_vgg(name)
                elif 'resnet101' == args.arch or 'resnet152' == args.arch:
                    image_size = calculate_image_size_bottlenet_resnet(name, image_size=64)
                elif 'resnet50' == args.arch:
                    image_size = calculate_image_size_imagenet_resnet(name, image_size=64)
                else:
                    raise EOFError(f"On tiny-imagenet-200, {args.arch} is not implemented !") 
            elif "cifar10" in data_path:
                if args.arch in ['resnet18Cifar', 'resnet50Cifar', 'resnet56Cifar']:
                    image_size = calculate_image_size_cifar10_resnet_basicblock(name)
                elif 'resnet101' == args.arch or 'resnet152' == args.arch:
                    image_size = calculate_image_size_bottlenet_resnet(name, image_size=32)
                elif 'vgg' in args.arch:
                    image_size = calculate_image_size_cifar10_vgg(name)
                elif 'mobilenetV2' == args.arch:
                    image_size = calculate_image_size_cifar10_mobilenet(name)
                elif 'shufflenetV2' == args.arch:
                    image_size = calculate_image_size_cifar10_shufflenet(name)
                else:
                    raise EOFError("On cifar10, args.arch is wrong !")
            else:
                raise EOFError("data_path is wrong !")
            
            flops_alive_per_layer = nz_count * 2 * image_size * image_size
            flops_origin_per_layer = total_params * 2 * image_size * image_size
            flops_total_alive += flops_alive_per_layer
            flops_total_origin += flops_origin_per_layer

            if "conv" in name:
                conv_flops_total_alive += flops_alive_per_layer
                conv_flops_total_origin += flops_origin_per_layer

            # print(f'{name:20} | alive_flops = {flops_alive_per_layer:7} / {flops_origin_per_layer:7} ({100 * flops_alive_per_layer / flops_origin_per_layer:6.2f}%) | '
            #       f'total_pruned = {flops_origin_per_layer - flops_alive_per_layer :7} | shape = {tensor.shape}|'
            #       f'image_size={image_size}')
    
    pruned = 100 * (flops_total_origin - flops_total_alive) / flops_total_origin
    flops_remain = 100 - pruned
    flops_speedup = flops_total_origin/flops_total_alive
    conv_pruned = 100 * (conv_flops_total_origin - conv_flops_total_alive) / conv_flops_total_origin
    print(f"===> flops: alive: {flops_total_alive}, pruned: {flops_total_origin-flops_total_alive}, total: {flops_total_origin}, "
          f"Speedup: {flops_speedup:10.2f}x, "
          f'({pruned:6.2f}% pruned, {flops_remain:6.2f}% remained; conv: {conv_pruned:6.2f}% pruned, {100 - conv_pruned:6.2f}% remained)')
    
    # sys.exit()
    return flops_speedup, flops_remain

def calculate_image_size_imagenet_resnet(name, image_size=224):
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


def calculate_image_size_bottlenet_resnet(name, image_size=64):
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

# for rn50
# def calculate_image_size_cifar10_rn50(name):
#     if name == "conv1.weight_prune":
#         size = 16

#     elif "layer1" in name:
#         size = 8
#     elif name == "layer2.0.conv1.weight_prune":
#         size = 8

#     elif "layer2" in name:
#         size = 4
#     elif name == "layer3.0.conv1.weight_prune":
#         size = 4

#     elif "layer3" in name:
#         size = 2
#     elif name == "layer4.0.conv1.weight_prune":
#         size = 2
    
#     elif "layer4" in name:
#         size = 1
    
#     elif "fc" in name:
#         size = 1
    
#     else:
#         print("name is wrong!")
#     return size

def load_weights_from_target_round(args, target_round):
    # load weights of the original model from the last epoch of round 0
    print(f"reset_weights to round: {target_round}")
    rewind_filename = f"{args.workspace}/pruningRound_{target_round}_checkpoint.pth.tar"
    # for test: rewind_filename = f"logs_rn50/resnet56Cifar_structureActivThresholdPruneNext_newAdaptiveRetrain70Accloss0.01_saveInterval10_normal_pruningRate5_trials1_bs128_epochs182_lr0.1_ngpu2_threshold0.0_rewind112_power1_cifar10/pruningRound_0_checkpoint.pth.tar"
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

def apdaptive_pruning_given_accloss_new(
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

    logging.basicConfig(filename=args.workspace + '/adaptive_threshold_given_accloss.log', level=logging.DEBUG)
    logging.info(file_thresholds_content)

    assert len(para_remained_per_list) == curr_round+1
    assert len(conv_threshold_list) == curr_round+1
    # acc_loss = acc_list[0] - acc_list[-1]

    # if curr_reduction <= target:
    #     lambda_value = lambda_list[-1]
    #     conv_threshold = conv_threshold_list[-1] + lambda_value
    #     track_rounds_list.append(curr_round)
    #     count_reset_time = 0
    #     # logging.info(f"acc_loss <= target; acc_loss={acc_loss}, curr_acc={acc_list[-1]}, target_acc={acc_list[0]}")
    #     logging.info(f"curr_reduction <= target; curr_reduction={curr_reduction}")
    if curr_reduction <= target:
        if one_condition:
            continue_flag = True
        else:
            assert target2 is not None and curr_reduction2 is not None
            # print(target2, curr_reduction2)
            if curr_reduction2 <= target2:
                continue_flag = True
        
        if continue_flag:
            lambda_value = lambda_list[-1]
            conv_threshold = conv_threshold_list[-1] + lambda_value
            track_rounds_list.append(curr_round)
            count_reset_time = 0
            # logging.info(f"acc_loss <= target; acc_loss={acc_loss}, curr_acc={acc_list[-1]}, target_acc={acc_list[0]}")
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
                
            # logging.info(f"acc_loss > target_acc_loss; reset to round: {index_back}; reset times: {count_reset_time}; after resetting, remained_per={remained_per_reset2}")
            logging.info(f"curr_reduction > target; reset to round: {index_back}; reset times: {count_reset_time}; after resetting, remained_per={remained_per_reset2}")

    logging.info(f"conv_threshold={conv_threshold}, lambda_value={lambda_value}, track_rounds_list: {track_rounds_list}\n")
    return lambda_value, conv_threshold, model_and_loss, track_rounds_list, count_reset_time 


# def apdaptive_pruning_given_accloss_new(
#     args, 
#     acc_list,
#     conv_remained_per_list,
#     conv_threshold_list,
#     lambda_list,
#     file_thresholds_content,
#     curr_round,
#     target_acc_loss,
#     model_and_loss,
#     track_rounds_list,
#     count_reset_time,
#     ):

#     logging.basicConfig(filename=args.workspace + '/adaptive_threshold_given_accloss.log', level=logging.DEBUG)
#     logging.info(file_thresholds_content)

#     assert len(conv_remained_per_list) == curr_round+1
#     assert len(conv_threshold_list) == curr_round+1
#     acc_loss = acc_list[0] - acc_list[-1]

#     if acc_loss <= target_acc_loss:
#         lambda_value = lambda_list[-1]
#         conv_threshold = conv_threshold_list[-1] + lambda_value
#         track_rounds_list.append(curr_round)
#         count_reset_time = 0
#         logging.info(f"acc_loss <= target_acc_loss; acc_loss={acc_loss}, curr_acc={acc_list[-1]}, target_acc={acc_list[0]}")
#     else:
#         if not track_rounds_list:
#             print("track_rounds_list is empty, exit.")
#             sys.exit()
#         else:
#             index_back = track_rounds_list[-1]

#             # reset model
#             loaded_model_state = load_weights_from_target_round(args, target_round=index_back)
#             model_and_loss.load_model_state(loaded_model_state)
#             remained_per_reset2, _, _, _ = print_nonzeros(model_and_loss.get_model(), curr_round, args)
            
#             # calculate lambda and threshold
#             count_reset_time = count_reset_time + 1
#             if count_reset_time == 3:
#                 lambda_value = lambda_list[-1]/2.0
#                 conv_threshold = conv_threshold_list[index_back]
#                 pop_index = track_rounds_list.pop()
#                 count_reset_time = 0
#                 assert pop_index == index_back
#             else:
#                 assert count_reset_time < 3
#                 lambda_value = lambda_list[index_back]/(2**count_reset_time)
#                 conv_threshold = conv_threshold_list[index_back] + lambda_value
                
#             logging.info(f"acc_loss > target_acc_loss; reset to round: {index_back}; reset times: {count_reset_time}; after resetting, remained_per={remained_per_reset2}")

#     logging.info(f"conv_threshold={conv_threshold}, lambda_value={lambda_value}, track_rounds_list: {track_rounds_list}\n")
#     return lambda_value, conv_threshold, model_and_loss, track_rounds_list, count_reset_time                                     