CUDA_VISIBLE_DEVICES=2 python ./multiproc.py --nproc_per_node 1 \
./main.py /cifar10_data --num-classes 10 \
--data-backend pytorch --raport-file raport.json -j8 -p 100 \
--workspace ${1:-./} --arch vgg19 -c fanin \
--label-smoothing 0.0 --lr-schedule step --lr_decay_epochs '150,180,210' --warmup 0 --mom 0.9 --wd 5e-4 \
--lr 0.05 --optimizer-batch-size 64 --batch-size 64 \
--prune_type structureActivThresholdPruneNext_givenAccLossParaReduce_newAdaptive_addThresholdWeightsPara_Retrain240AccLoss0.0012ParaReduce0.8Lambda1.0 \
--conv_threshold 0.0 --pruning_rate 5 \
--num_pruningRound 100 --epochs 240 --rewind_epoch 0 --target_accloss 0.0012 --target_parameters_reduce 0.8 --target_flops_reduce 0.0 --lambda_value 1.0 \
--power_value 1 \
--resume "./logs_rn50/cifar10_vgg19withBias_CRDSettting_original/best_pruningRound_0_checkpoint.pth.tar" # vgg19