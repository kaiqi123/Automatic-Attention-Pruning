CUDA_VISIBLE_DEVICES=1,2 python ./multiproc.py --nproc_per_node 2 \
./main.py /tiny-imagenet-200 --num-classes 200 \
--data-backend dali-cpu --raport-file raport.json -j8 -p 100 \
--workspace ${1:-./} --arch vgg19 -c fanin \
--label-smoothing 0.0 --lr-schedule step --lr_decay_epochs '150,225' --warmup 0 --mom 0.9 --wd 2e-4 \
--lr 0.1 --optimizer-batch-size 128 --batch-size 64 \
--prune_type structureActivThresholdPruneNext_givenFlopsReduce_newAdaptive_addThresholdWeightsFlops_Retrain300FlopsReduce0.67Lambda1.0 \
--conv_threshold 0.0 --pruning_rate 5 \
--num_pruningRound 100 --epochs 300 --rewind_epoch 0 --target_accloss 0.0 --target_parameters_reduce 0.0 --target_flops_reduce 0.67 --lambda_value 1.0 \
--power_value 1 \
--resume "./logs_rn50/tinyImageNet_vgg19_original/best_pruningRound_0_checkpoint.pth.tar"