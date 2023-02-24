# bkp command: givenParaReduce, givenFlopsReduce, givenAccLoss
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./multiproc.py --nproc_per_node 4 \
./main.py /cifar10_data --num-classes 10 \
--data-backend pytorch --raport-file raport.json -j8 -p 100 \
--workspace ${1:-./} --arch resnet50Cifar -c fanin \
--label-smoothing 0.0 --lr-schedule cifar10_rn18_step --warmup 0 --mom 0.9 --wd 0.0002 \
--lr 0.1 --optimizer-batch-size 512 --batch-size 128 \
--prune_type structureActivThresholdPruneNext_givenParaReduce_newAdaptive_addThresholdWeightsPara_Retrain182ParaReduce0.61Lambda1.0 \
--conv_threshold 0.0 --pruning_rate 5 \
--num_pruningRound 100 --epochs 182 --rewind_epoch 0 --target_accloss 0.0 --target_parameters_reduce 0.61 --target_flops_reduce 0.0 --lambda_value 1.0 \
--power_value 1 