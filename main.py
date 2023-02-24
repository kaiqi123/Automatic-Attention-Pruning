
import argparse
import os
import shutil
import time
import random
import sys

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to run this example."
    )

# import image_classification.resnet as models
# import image_classification.logger as log

from image_classification.smoothing import LabelSmoothing
from image_classification.mixup import NLLMultiLabelSmooth, MixUpWrapper
from image_classification.dataloaders import *
from image_classification.training import *
from image_classification.utils import *

import dllogger

from pruning import pruning_utils
from image_classification import customized_utils


def add_parser_arguments(parser):
    # model_names = models.resnet_versions.keys()
    # model_names = list(models.resnet_versions.keys()) + list(models_cifar10.resnet_versions.keys()) #new added
    # model_configs = models.resnet_configs.keys()
    # print(f"model_names: {model_names}")

    parser.add_argument("data", metavar="DIR", help="path to dataset")
    parser.add_argument(
        "--data-backend",
        metavar="BACKEND",
        default="dali-cpu",
        choices=DATA_BACKEND_CHOICES,
        help="data backend: "
        + " | ".join(DATA_BACKEND_CHOICES)
        + " (default: dali-cpu)",
    )

    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="resnet50",
        # choices=model_names,
        # help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
    )

    parser.add_argument(
        "--model-config",
        "-c",
        metavar="CONF",
        default="classic",
        # choices=model_configs,
        # help="model configs: " + " | ".join(model_configs) + "(default: classic)",
    )

    parser.add_argument(
        "--num-classes",
        metavar="N",
        default=1000,
        type=int,
        help="number of classes in the dataset",
    )

    parser.add_argument(
        "-j",
        "--workers",
        default=5,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 5)",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--run-epochs",
        default=-1,
        type=int,
        metavar="N",
        help="run only N epochs, used for checkpointing runs",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256) per gpu",
    )

    parser.add_argument(
        "--optimizer-batch-size",
        default=-1,
        type=int,
        metavar="N",
        help="size of a total batch size, for simulating bigger batches using gradient accumulation",
    )

    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument(
        "--lr-schedule",
        default="step",
        type=str,
        metavar="SCHEDULE",
        choices=["step", "linear", "cosine", "cifar10_rn18_step", "cifar10_vgg_step"],
        help="Type of LR schedule: {}, {}, {}".format("step", "linear", "cosine"),
    )

    parser.add_argument(
        "--warmup", default=0, type=int, metavar="E", help="number of warmup epochs"
    )

    parser.add_argument(
        "--label-smoothing",
        default=0.0,
        type=float,
        metavar="S",
        help="label smoothing",
    )
    parser.add_argument(
        "--mixup", default=0.0, type=float, metavar="ALPHA", help="mixup alpha"
    )

    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--bn-weight-decay",
        action="store_true",
        help="use weight_decay on batch normalization learnable parameters, (default: false)",
    )
    parser.add_argument(
        "--nesterov",
        action="store_true",
        help="use nesterov momentum, (default: false)",
    )

    parser.add_argument(
        "--print-freq",
        "-p",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--pretrained-weights",
        default="",
        type=str,
        metavar="PATH",
        help="load weights from here",
    )

    parser.add_argument("--fp16", action="store_true", help="Run model fp16 mode.")
    parser.add_argument(
        "--static-loss-scale",
        type=float,
        default=1,
        help="Static loss scale, positive power of 2 values can improve fp16 convergence.",
    )
    parser.add_argument(
        "--dynamic-loss-scale",
        action="store_true",
        help="Use dynamic loss scaling.  If supplied, this argument supersedes "
        + "--static-loss-scale.",
    )
    parser.add_argument(
        "--prof", type=int, default=-1, metavar="N", help="Run only N iterations"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Run model AMP (automatic mixed precision) mode.",
    )

    parser.add_argument(
        "--seed", default=None, type=int, help="random seed used for numpy and pytorch"
    )

    parser.add_argument(
        "--gather-checkpoints",
        action="store_true",
        help="Gather checkpoints throughout the training, without this flag only best and last checkpoints will be stored",
    )

    parser.add_argument(
        "--raport-file",
        default="experiment_raport.json",
        type=str,
        help="file in which to store JSON experiment raport",
    )

    parser.add_argument(
        "--evaluate", action="store_true", help="evaluate checkpoint/model"
    )
    parser.add_argument("--training-only", action="store_true", help="do not evaluate")

    parser.add_argument(
        "--no-checkpoints",
        action="store_false",
        dest="save_checkpoints",
        help="do not store any checkpoints, useful for benchmarking",
    )

    parser.add_argument("--checkpoint-filename", default="checkpoint.pth.tar", type=str)
    
    parser.add_argument(
        "--workspace",
        type=str,
        default="./",
        metavar="DIR",
        help="path to directory where checkpoints will be stored",
    )
    parser.add_argument(
        "--memory-format",
        type=str,
        default="nchw",
        choices=["nchw", "nhwc"],
        help="memory layout, nchw or nhwc",
    )

    # PruningHparams
    parser.add_argument(
        "--prune_type",
        default="normal",
        type=str,
        help="nonstructure_normal | structureActiv_normal |"
             " structureL1norm_normal | structureActivThreshold_normal",
    )
    parser.add_argument("--lr_decay_epochs", default='', type=str)
    parser.add_argument('--lr_decay_rate', default=0.1, type=float, help='decay rate for learning rate')
    parser.add_argument("--num_pruningRound", default=25, type=int)
    parser.add_argument("--conv_threshold", default=-1.0, type=float)
    parser.add_argument("--rewind_epoch", default=-1, type=int)
    parser.add_argument("--num_trails", default=1, type=int)
    parser.add_argument("--power_value", default=-1, type=int)
    parser.add_argument("--pruning_rate", default=20, type=int)
    parser.add_argument("--target_accloss", default=0.0, type=float)
    parser.add_argument("--target_parameters_reduce", default=0.0, type=float)
    parser.add_argument("--target_flops_reduce", default=0.0, type=float)
    parser.add_argument("--lambda_value", default=0.1, type=float)


def main(args):
    global best_prec1
    best_prec1 = 0

    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
        args.local_rank = int(os.environ["LOCAL_RANK"])
        print("WORLD_SIZE in os.environ", args.distributed, args.local_rank)

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend="nccl", init_method="env://")
        # dist.init_process_group(backend="gloo", init_method="env://")
        args.world_size = torch.distributed.get_world_size()
        print("args.distributed is True", args.gpu, args.world_size, args.local_rank)

    if args.amp and args.fp16:
        print("Please use only one of the --fp16/--amp flags")
        exit(1)

    print(args.gpu, args.world_size, args.amp, args.fp16)
    

    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed + args.local_rank)
        torch.cuda.manual_seed(args.seed + args.local_rank)
        np.random.seed(seed=args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)

        def _worker_init_fn(id):
            np.random.seed(seed=args.seed + args.local_rank + id)
            random.seed(args.seed + args.local_rank + id)

    else:
        def _worker_init_fn(id):
            pass

    if args.fp16:
        assert (
            torch.backends.cudnn.enabled
        ), "fp16 mode requires cudnn backend to be enabled."

    if args.static_loss_scale != 1.0:
        if not args.fp16:
            print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")

    if args.optimizer_batch_size < 0:
        batch_size_multiplier = 1
    else:
        tbs = args.world_size * args.batch_size
        if args.optimizer_batch_size % tbs != 0:
            print(
                "Warning: simulated batch size {} is not divisible by actual batch size {}".format(
                    args.optimizer_batch_size, tbs
                )
            )
        batch_size_multiplier = int(args.optimizer_batch_size / tbs)
        print("BSM: {}".format(batch_size_multiplier))
    print(args.world_size, args.batch_size, args.optimizer_batch_size, tbs, batch_size_multiplier)
    # sys.exit()

    pretrained_weights = None
    if args.pretrained_weights:
        if os.path.isfile(args.pretrained_weights):
            print("=> loading pretrained weights from '{}'".format(args.pretrained_weights))
            pretrained_weights = torch.load(args.pretrained_weights)
        else:
            print("=> no pretrained weights found at '{}'".format(args.resume))

    start_epoch = 0
    # optionally resume from a checkpoint
    if args.resume is not None:
        # args.resume = args.workspace + "/" + args.resume
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}', args.gpu: {}".format(args.resume, args.gpu))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu))
            start_epoch = checkpoint["epoch"]
            best_prec1 = checkpoint["best_prec1"]
            model_state = checkpoint["state_dict"]
            optimizer_state = checkpoint["optimizer"]
            print("=> loaded checkpoint '{}' (epoch {}), best_prec1: {}, optimizer_state: {}".
                  format(args.resume, start_epoch, best_prec1, optimizer_state.keys()))
        else:
            raise EOFError("=> no checkpoint found at '{}'".format(args.resume))

    else:
        model_state = None
        optimizer_state = None
    # sys.exit()

    loss = nn.CrossEntropyLoss
    if args.mixup > 0.0:
        loss = lambda: NLLMultiLabelSmooth(args.label_smoothing)
    elif args.label_smoothing > 0.0:
        loss = lambda: LabelSmoothing(args.label_smoothing)
    # print(args.mixup, args.label_smoothing)

    memory_format = (torch.channels_last if args.memory_format == "nhwc" else torch.contiguous_format)
    # print(args.memory_format, memory_format)

    model_and_loss = ModelAndLoss(
        (args.arch, args.model_config, args.num_classes),
        loss,
        pretrained_weights=pretrained_weights,
        cuda=True,
        fp16=args.fp16,
        memory_format=memory_format,
        mask=True,
        # batch_size_for_relu=256,
        data_path = args.data,
    )
    # print(1111, args.arch, args.model_config, args.num_classes)

    # Create data loaders and optimizers as needed
    if args.data_backend == "pytorch":
        get_train_loader = get_pytorch_train_loader
        get_val_loader = get_pytorch_val_loader
    elif args.data_backend == "dali-gpu":
        get_train_loader = get_dali_train_loader(dali_cpu=False)
        get_val_loader = get_dali_val_loader()
    elif args.data_backend == "dali-cpu":
        get_train_loader = get_dali_train_loader(dali_cpu=True)
        get_val_loader = get_dali_val_loader()
    elif args.data_backend == "syntetic":
        get_val_loader = get_syntetic_loader
        get_train_loader = get_syntetic_loader

    train_loader, train_loader_len = get_train_loader(
        args.data,
        args.batch_size,
        args.num_classes,
        args.mixup > 0.0,
        start_epoch=start_epoch,
        workers=args.workers,
        fp16=args.fp16,
        memory_format=memory_format,
    )
    if args.mixup != 0.0:
        train_loader = MixUpWrapper(args.mixup, train_loader)
    
    val_loader, val_loader_len = get_val_loader(
        args.data,
        args.batch_size,
        args.num_classes,
        False,
        workers=args.workers,
        fp16=args.fp16,
        memory_format=memory_format,
    )
    print(f"args.workers: {args.workers}")
    # sys.exit()

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger = log.Logger(
            args.print_freq,
            [
                dllogger.StdOutBackend(
                    dllogger.Verbosity.DEFAULT, step_format=log.format_step
                ),
                dllogger.JSONStreamBackend(
                    dllogger.Verbosity.VERBOSE,
                    os.path.join(args.workspace, args.raport_file),
                ),
            ],
            start_epoch=start_epoch - 1,
        )
    else:
        logger = log.Logger(args.print_freq, [], start_epoch=start_epoch - 1)
    logger.log_parameter(args.__dict__, verbosity=dllogger.Verbosity.DEFAULT)

    named_parameters = list(model_and_loss.model.named_parameters())
    trainable_parameters = [(n, v) for n, v in named_parameters if v.requires_grad]
    print("length of total parameters (260): {}, length of trainable parameters (161): {}".format(len(named_parameters), len(trainable_parameters)))
   
    optimizer = get_optimizer(
        trainable_parameters,
        args.fp16,
        args.lr,
        args.momentum,
        args.weight_decay,
        nesterov=args.nesterov,
        bn_weight_decay=args.bn_weight_decay,
        state=optimizer_state,
        static_loss_scale=args.static_loss_scale,
        dynamic_loss_scale=args.dynamic_loss_scale,
        # load_state_flag=False if optimizer_state is None else True, #new added, for define optimizer in each step
    )
    # print("optimizer_state: {}".format(optimizer_state.keys()))

    if args.lr_schedule == "step":
        # lr_policy = lr_step_policy(args.lr, [30, 60, 80], 0.1, args.warmup, logger=logger) # org
        # lr_policy = lr_step_policy(args.lr, [91, 136], 0.1, args.warmup, logger=logger) # cifar10_rn18_step
        assert args.lr_decay_epochs != ''
        lr_decay_epochs = [int(e) for e in list(args.lr_decay_epochs.split(","))]
        lr_policy = lr_step_policy(args.lr, lr_decay_epochs, args.lr_decay_rate, args.warmup, logger=logger)
        print("===> Learning rate decay epochs:",type(lr_decay_epochs), lr_decay_epochs)
    elif args.lr_schedule == "cosine":
        lr_policy = lr_cosine_policy(args.lr, args.warmup, args.epochs, logger=logger)
    elif args.lr_schedule == "linear":
        lr_policy = lr_linear_policy(args.lr, args.warmup, args.epochs, logger=logger)
    else:
        raise EOFError("args.lr_schedule is wrong!")       
        
    
    if args.amp:
        model_and_loss, optimizer = amp.initialize(
            model_and_loss,
            optimizer,
            opt_level="O1",
            loss_scale="dynamic" if args.dynamic_loss_scale else args.static_loss_scale,
        )

    if args.distributed:
        model_and_loss.distributed()

    # Note: load_model_state() changed
    model_and_loss.load_model_state(model_state)
    
    # Original trainable parameters
    # resnet50: full-imagenet-rn50: 25557032; tiny-200-32: rn50: 23917832
    # resnet34: full-imagenet-rn50: 21797672; tiny-200-32: rn50: 23917832
    # resnet18_cifar10: 272474; resnet56_cifar10: 855770; resnet50_cifar10: 23528522
    # tiny-imagenet: resnet-101: 42909960; resnet-152: 58553608
    num_trainable_parameters = sum(p.numel() for p in model_and_loss.model.parameters() if p.requires_grad)
    print(f"num_trainable_parameters: {num_trainable_parameters}")
    # sys.exit()
    
    start_pruningRound = 0
    rewind_filename = f"rewindEpoch_{args.rewind_epoch}_round_0_checkpoint.pth.tar"
    num_pruningRound = args.num_pruningRound
    resample = True if "resample" in args.prune_type else False
    reinit = True if "reinit" in args.prune_type else False

    # for calculating threshold
    conv_threshold = 0.0 
    lambda_value = args.lambda_value # 0.1 for cifar10, 0.005 for imagenet
    para_remained_per_list = []
    metric_list = []
    acc_list = []
    conv_threshold_list = []
    lambda_list = []
    flops_remained_list = []
    minus_threshold_flag = False
    target_acc_loss = args.target_accloss * 100 if "givenAccLoss" in args.prune_type or "givenAccLossParaReduce" in args.prune_type else None 
    target_parameters_reduce = args.target_parameters_reduce * 100 if "givenParaReduce" in args.prune_type  or "givenAccLossParaReduce" in args.prune_type else None
    target_flops_reduce = args.target_flops_reduce * 100 if "givenFlopsReduce" in args.prune_type else None
    fix_conv_threshold = args.conv_threshold if "fixThreshold" in args.prune_type else None
    
    track_rounds_list = []  # track all the rounds where accurayc, parameters, flops are within the target
    count_reset_time = 0
    print(f"fix_conv_threshold: {fix_conv_threshold}, lambda_value: {lambda_value}; target_acc_loss: {target_acc_loss}; target_parameters_reduce: {target_parameters_reduce}; target_flops_reduce: {target_flops_reduce}")
    # sys.exit()
    # fake_acc = [92.0, 91.0, 91.0, 91.0, 90.0, 91.0, 90, 90, 91, 90, 90, 90, 91, 85, 85, 85, 85, 85, 91]
    # fake_acc = [92.0, 91.0, 91.0, 91.0, 90.0, 91.0, 90, 90, 91, 90, 90, 90, 85, 85, 85, 85, 85, 85, 91]
    # fake_acc = [92.7885, 93.0088, 93.2392, 93.3694, 92.4179, 91.1558, 92.3578, 92.0473, 91.1659, 91.1959, 91.5064, 91.9571, 91.4263, 91.5865, 91.7368, 91.7969, 91.9071, 91.9671]
    # fake_remain_para = [100, 95.0, 95.0, 95.0, 75.0, 95.0, 75, 75, 95, 75, 75, 75, 75, 75, 75, 75, 75, 75, 95]
    # fake_remain_para = [100, 100.0, 100.0, 100.0, 100.0, 100.0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    # fake_remain_flops = [100, 95.0, 95.0, 95.0, 75.0, 95.0, 75, 75, 95, 75, 75, 75, 75, 75, 75, 75, 75, 75, 95]

    for curr_round in range(start_pruningRound, num_pruningRound):
        print("\n\ncurr_round: {}".format(curr_round))
        round_start_time = time.time()
        if not curr_round == 0:
            # prune
            # conv_threshold, std_list = pruning_utils.prune(
            metric_list = pruning_utils.prune(
                curr_round,
                prune_type=args.prune_type,
                model=model_and_loss.get_model(),
                model_name=args.arch,
                resample=resample,
                reinit=reinit,
                args=args,
                conv_threshold=conv_threshold,
                power_value=args.power_value,
                # conv_threshold_list=conv_threshold_list,
                # conv_remained_per_list=para_remained_per_list,
                # acc_list = acc_list,
                # std_list = std_list,
                # conv_threshold_dict=conv_threshold_dict,
                )

            # load rewinding weights and optimizer
            # loaded_optimizer_state_dict, loaded_model_state = pruning_utils.load_weights(
            #     gpu=args.gpu,
            #     rewind_epoch=args.rewind_epoch,
            #     rewind_filename=os.path.join(args.workspace, rewind_filename))

            # load weights of the original model from the last epoch of round 0
            # loaded_optimizer_state_dict, loaded_model_state = pruning_utils.load_weights(
            #     gpu = args.gpu,
            #     rewind_epoch = 49,
            #     rewind_filename = "./logs_rn50/tinyImageNet_bs256_epochs50_lr0.1024_pruningRound_0_checkpoint.pth.tar")

            # rewind weights
            # pruning_utils.rewind_weights(model=model_and_loss.get_model(), state_dict=loaded_model_state)

            # Only rewind learning rate
            # get a new optimizer, note: state=None (not load optimizer state), otherwise parameters cannot reduce
            trainable_parameters = [(n, v) for n, v in list(model_and_loss.get_model().named_parameters())  if v.requires_grad]
            optimizer = get_optimizer(
                trainable_parameters,
                args.fp16,
                args.lr,
                args.momentum,
                args.weight_decay,
                nesterov=args.nesterov,
                bn_weight_decay=args.bn_weight_decay,
                state=None,
                static_loss_scale=args.static_loss_scale,
                dynamic_loss_scale=args.dynamic_loss_scale,
                # load_state_flag = False,
            )

            # check optimizer state
            # pruning_utils.check_optimizer_state(optimizer.state_dict())


        TRIAL_NUM = 0
        print(f"--- Pruning Level [{TRIAL_NUM}:{curr_round}/{num_pruningRound}]: {1.0 * (0.8 ** (curr_round)) * 100:.1f}---")
        
        # calcualte number of parameters
        remained_per, remained_fc, remained_conv, compress_ratio = pruning_utils.print_nonzeros(model_and_loss.get_model(), curr_round, args)

        # calcualte flops
        flops_speedup, flops_remain = pruning_utils.calculate_flops(model_and_loss.get_model(), args.data, args)
        
        # start_epoch = args.epochs - 1 # for test
        start_epoch = start_epoch if curr_round == 0 else args.rewind_epoch + 1 # formal, both cifar10 and imagenet

        end_epoch = args.epochs

        checkpoint_filename = f"pruningRound_{curr_round}_"+args.checkpoint_filename
        accuracy = train_loop(
            model_and_loss,
            optimizer,
            lr_policy,
            train_loader,
            val_loader,
            args.fp16,
            logger,
            should_backup_checkpoint(args),
            use_amp=args.amp,
            batch_size_multiplier=batch_size_multiplier,
            start_epoch=start_epoch,
            # end_epoch=(start_epoch + args.run_epochs) if args.run_epochs != -1 else args.epochs,
            end_epoch=end_epoch,
            best_prec1=best_prec1,
            prof=args.prof,
            skip_training=args.evaluate,
            skip_validation=args.training_only,
            save_checkpoints=args.save_checkpoints and not args.evaluate,
            checkpoint_dir=args.workspace,
            checkpoint_filename=checkpoint_filename,
            rewind_filename=rewind_filename if curr_round == 0 else None,
            rewind_epoch=args.rewind_epoch,
            log_filename=args.workspace + f"/log/pruningRound_{curr_round}.log",
            remained_per=remained_per, # for saving to log.txt
            compress_ratio=compress_ratio, # for saving to log.txt
            flops_speedup=flops_speedup, # for saving to log.txt
            flops_remain=flops_remain, # for saving to log.txt
            args = args, # for saving weights of the last round
            curr_round=curr_round, # for saving weights of the last round
            # loaded_optimizer_state_dict=None if curr_round==0 else loaded_optimizer_state_dict, # for define optimizer in every step
        )
        round_duration = time.time() - round_start_time
        round_duration = round(round_duration / 3600.0, 4)
        print(f"Training time per round: {round_duration}h") # 7.86h

        # save threshold analysis
        if "ActivThreshold" in args.prune_type:
            
            # accuracy = fake_acc[curr_round]
            # remained_per = fake_remain_para[curr_round]
            # flops_remain = fake_remain_flops[curr_round]
            
            acc_list.append(accuracy)
            para_remained_per_list.append(remained_per)
            conv_threshold_list.append(conv_threshold)
            lambda_list.append(lambda_value)
            flops_remained_list.append(flops_remain)
            
            # save to thresholds.log
            file_thresholds_path = args.workspace + f"/thresholds.log"
            std_value = -1.0 if curr_round == 0 else metric_list[0]
            mean_value = -1.0 if curr_round == 0 else metric_list[1]
            file_thresholds_content = f"PruningRound = {curr_round}\tAccuracy = {round(accuracy,4)}\tremained_per = {remained_per}\tremained_flops = {flops_remain}\t"\
                        +f"conv_threshold = {round(float(conv_threshold),4)}\tlambda_value = {round(float(lambda_value),4)}\t"\
                        +f"std = {round(float(std_value),4)}\tmean = {round(float(mean_value),4)}\ttime_per_round = {round_duration}"
            write_to_log(file_thresholds_path, file_thresholds_content)
            # print(file_thresholds_content)
            
            # Given target_acc_loss, calculate threshold
            if "_givenAccLoss_" in args.prune_type:
                target = target_acc_loss
                curr_reduction =   acc_list[0] - acc_list[-1]

            elif "_givenParaReduce_" in args.prune_type:
                target = target_parameters_reduce
                curr_reduction = para_remained_per_list[0] - para_remained_per_list[-1]

            elif "_givenFlopsReduce_" in args.prune_type:
                target = target_flops_reduce
                curr_reduction = flops_remained_list[0] - para_remained_per_list[-1]

            elif "_givenAccLossParaReduce_" in args.prune_type:
                target = target_acc_loss
                curr_reduction =   acc_list[0] - acc_list[-1]
                one_condition = False
                target2 = target_parameters_reduce
                curr_reduction2 = para_remained_per_list[0] - para_remained_per_list[-1]

            else:
                raise EOFError(f"{args.prune_type} is not implemented!")
            
            lambda_value, conv_threshold, model_and_loss, track_rounds_list, count_reset_time = pruning_utils.apdaptive_pruning_given_accloss_new(
                args = args, 
                acc_list = acc_list,
                para_remained_per_list = para_remained_per_list,
                flops_remained_list = flops_remained_list,
                conv_threshold_list = conv_threshold_list,
                lambda_list = lambda_list,
                file_thresholds_content = file_thresholds_content,
                curr_round = curr_round,
                model_and_loss = model_and_loss,
                track_rounds_list = track_rounds_list,
                count_reset_time = count_reset_time,
                target = target, # difference among three methods
                curr_reduction = curr_reduction, # difference among three methods
                one_condition = one_condition, # difference among one and two conditions
                target2 = target2, # difference among one and two conditions
                curr_reduction2 = curr_reduction2, # difference among one and two conditions
            )

            # do inference
            print("Do inference for 1 epoch")
            _ = train_loop(
                model_and_loss,
                optimizer,
                lr_policy,
                train_loader,
                val_loader,
                args.fp16,
                None,
                should_backup_checkpoint(args),
                use_amp=args.amp,
                batch_size_multiplier=None,
                start_epoch=0,
                end_epoch=1,
                best_prec1=None,
                prof=args.prof,
                skip_training=True,
                skip_validation=False,
                save_checkpoints=None,
                checkpoint_dir=None,
                checkpoint_filename=None,
                rewind_filename=None,
                rewind_epoch=None,
                log_filename=None,
                remained_per=None, # for saving to log.txt
                compress_ratio=None, # for saving to log.txt
                flops_speedup=None, # for saving to log.txt
                flops_remain=None, # for saving to log.txt
                args = None, # for saving weights of the last round
                curr_round=None, # for saving weights of the last round
                # loaded_optimizer_state_dict=None if curr_round==0 else loaded_optimizer_state_dict, # for define optimizer in every step
            )

        if remained_per < 1.0:
            print("Note: remained_per < 1.0 !")
            break

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.end()


if __name__ == "__main__": 
    exp_start_time = time.time()

    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    add_parser_arguments(parser)
    args = parser.parse_args()
    cudnn.benchmark = True
    
    args.workspace = "./logs_rn50/cifar10_vgg19_twoConditions/" + f"{args.arch}_{args.prune_type}_pruningRate{args.pruning_rate}_trials{args.num_trails}_bs{args.batch_size}_" \
            f"epochs{args.epochs}_lr{args.lr}_ngpu{torch.cuda.device_count()}_threshold{args.conv_threshold}_rewind{args.rewind_epoch}_"\
            f"power{args.power_value}"
    # args.workspace = "./logs_rn50/tinyImageNet_resnet50/" + f"{args.arch}_{args.prune_type}_pruningRate{args.pruning_rate}_trials{args.num_trails}_bs{args.batch_size}_" \
    #         f"epochs{args.epochs}_lr{args.lr}_ngpu{torch.cuda.device_count()}_threshold{args.conv_threshold}_rewind{args.rewind_epoch}_"\
    #         f"power{args.power_value}"
    # args.workspace = "./logs_rn50/cifar10_resnet152/" + f"{args.arch}_{args.prune_type}_pruningRate{args.pruning_rate}_trials{args.num_trails}_bs{args.batch_size}_" \
    #         f"epochs{args.epochs}_lr{args.lr}_ngpu{torch.cuda.device_count()}_threshold{args.conv_threshold}_rewind{args.rewind_epoch}_"\
    #         f"power{args.power_value}"
    # args.workspace = "./logs_rn50/imagenet_resnet50_logs0913/" + f"{args.arch}_{args.prune_type}_pruningRate{args.pruning_rate}_trials{args.num_trails}_bs{args.batch_size}_" \
    #         f"epochs{args.epochs}_lr{args.lr}_ngpu{torch.cuda.device_count()}_threshold{args.conv_threshold}_rewind{args.rewind_epoch}_"\
    #         f"power{args.power_value}"
    # args.workspace = "./logs_rn50/" + "test"
    if not os.path.exists(args.workspace+"/log"):
        os.makedirs(args.workspace+"/log")
    print("workspace: {}".format(args.workspace))

    main(args)

    exp_duration = time.time() - exp_start_time
    print("Experiment ended, exp_duration: {}h".format(round(exp_duration/3600.0, 4)))
    # print("Experiment ended, exp_duration: {}s".format(round(exp_duration, 2)))


