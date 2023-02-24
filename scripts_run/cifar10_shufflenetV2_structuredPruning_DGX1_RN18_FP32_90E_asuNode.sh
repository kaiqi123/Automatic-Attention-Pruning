CUDA_VISIBLE_DEVICES=3 python ./multiproc.py --nproc_per_node 1 \
./main.py /cifar10_data --num-classes 10 \
--data-backend pytorch --raport-file raport.json -j8 -p 100 \
--workspace ${1:-./} --arch shufflenetV2 -c fanin \
--label-smoothing 0.0 --lr-schedule step --lr_decay_epochs '60,120,160' --warmup 0 --mom 0.9 --wd 5e-4 --lr_decay_rate 0.2 \
--lr 1e-1 --optimizer-batch-size 128 --batch-size 128 \
--prune_type structureActivThresholdPruneNext_givenAccLoss_newAdaptive_addThresholdWeightsPara_Retrain200AccLoss0.0031Lambda1.0_noConv3Relu \
--conv_threshold 0.0 --pruning_rate 5 \
--num_pruningRound 100 --epochs 200 --rewind_epoch 0 --target_accloss 0.0031 --target_parameters_reduce 0.0 --target_flops_reduce 0.0 --lambda_value 1.0 \
--power_value 1


# --resume "./logs_rn50/cifar10_mobilenetV2_original/best_pruningRound_0_checkpoint.pth.tar"