
CUDA_VISIBLE_DEVICES=2,3 python ./multiproc.py --nproc_per_node 2 \
./main.py /tiny-imagenet-200 --num-classes 200 \
--data-backend dali-cpu --raport-file raport.json -j8 -p 100 \
--workspace ${1:-./} --arch resnet101 -c fanin \
--label-smoothing 0.0 --lr-schedule step --lr_decay_epochs '150,225' --warmup 0 --mom 0.9 --wd 2e-4 \
--lr 0.1 --optimizer-batch-size 512 --batch-size 256 \
--prune_type structureActivThresholdPruneNext_givenParaReduce_newAdaptive_addThresholdWeightsPara_Retrain300ParaReduce0.76Lambda10.0 \
--conv_threshold 0.0 --pruning_rate 5 \
--num_pruningRound 100 --epochs 300 --rewind_epoch 0 --target_accloss 0.0 --target_parameters_reduce 0.76 --target_flops_reduce 0.0 --lambda_value 10.0 \
--power_value 1 \
--resume "./logs_rn50/tinyImageNet_resnet101_original/best_pruningRound_0_checkpoint.pth.tar"