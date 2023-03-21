# ============================================ CIFAR-10, ResNet-56 ==================================================================
# The choices of prune_type are as follows:
# AAP_givenAccLoss_minimizePara
# AAP_givenAccLoss_minimizeFlops
# AAP_givenParaReduce
# AAP_givenFlopsReduce
# AAP_givenAccLossParaReduce

# Remember to change to following 2 lines in the model_module.py of the folder "pruning" as needed
# curr_conv_threshold = conv_threshold * threshold_weights_dict_para[name] # set threshold using weights calculated by parameters
# curr_conv_threshold = conv_threshold * threshold_weights_dict_flops[name] # set threshold using weights calculated by flops
# ====================================================================================================================================

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./multiproc.py --nproc_per_node 4 \
./main.py /cifar10_data --num-classes 10 \
--data-backend pytorch --raport-file raport.json -j8 -p 100 \
--workspace ${1:-./} --arch resnet56Cifar -c fanin \
--label-smoothing 0.0 --lr-schedule step --lr_decay_epochs '91,136' --warmup 0 --mom 0.9 --wd 0.0002 \
--lr 0.1 --optimizer-batch-size 512 --batch-size 128 \
--prune_type AAP_givenAccLoss_minimizePara \
--num_pruningRound 100 --epochs 182 --rewind_epoch 0 --target_accloss 0.0 --target_parameters_reduce 0.0 --target_flops_reduce 0.53 --lambda_value 1.0 \
--power_value 1
