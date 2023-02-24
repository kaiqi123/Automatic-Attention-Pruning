CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ./multiproc.py --nproc_per_node 4 \
./main.py /imagenet_data --num-classes 1000 \
--data-backend dali-cpu --raport-file raport.json -j4 -p 100 \
--workspace ${1:-./} --arch resnet50 -c fanin \
--label-smoothing 0.1 --lr-schedule step --warmup 5 --mom 0.9 --wd 0.0001 \
--lr 0.256 --optimizer-batch-size 256 --batch-size 64 \
--prune_type structureActivThresholdPruneNext_givenAccLoss_newAdaptive_addThresholdWeightsPara_Retrain90AccLoss0.002Lambda0.1 \
--conv_threshold 0.0 --pruning_rate 5 \
--num_pruningRound 100 --epochs 90 --rewind_epoch 0 --target_accloss 0.002 --target_parameters_reduce 0.0 --target_flops_reduce 0.0 --lambda_value 0.1 \
--power_value 1 \
--resume "./logs_rn50/imageNet_resnet50_original/imageNet_resnet50_best_pruningRound_0_checkpoint.pth.tar"