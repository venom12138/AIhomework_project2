export project_name=AIproject2
export exp_name=11_22baseline
export exp_args="--dataset emotiondataset --randaugment --cos_lr" # 一组实验很多个run
export branch=master

#export cluster=RTX2080Ti
# export gpus=1

# network=resnet
# for layers in {32,44,110}; do
#   # for aux_config in {0c1f,0c2f,1c1f,1c2f,1c3f,2c2f}; do
#   export run_name="network=$network,layers=$layers"
#   bash base.sh --model $network --layers $layers &
#   sleep 2
#   # done 
# done
# wait

network=wideresnet
for layers in {16,28,52}; do
  # for aux_config in {0c1f,0c2f,1c1f,1c2f,1c3f,2c2f}; do
  export run_name="network=$network,layers=$layers"
  bash base.sh --model $network --layers $layers &
  sleep 2
  # done
done
wait

# network=densenet_bc
# for layers in {121,169,201}; do
#   # for aux_config in {0c1f,0c2f,1c1f,1c2f,1c3f,2c2f}; do
#   export run_name="network=$network,layers=$layers"
#   bash base.sh --model $network --layers $layers &
#   sleep 2
#   # done
# done
# wait

network=resnext
for cardinality in {16，}; do
  # for aux_config in {0c1f,0c2f,1c1f,1c2f,1c3f,2c2f}; do
  export run_name="network=$network,cardinality=$cardinality"
  bash base.sh --model $network --cardinality $cardinality &
  sleep 2
  # done
done
wait