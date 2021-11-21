export WANDB_PROJECT=11_21train

cd "/home/yu-jw19/venom/project2"

# [ -z "$commit" ] && commit=$branch # -z 判断 变量的值，是否为空；
# git checkout $commit
# git reset --hard HEAD

# [ -d data ] || ln -s /cluster/home/nzl/1025_cifar_venom/data_new data # 或操作执行前面那个
# # -d 如果data目录存在
# export commit=$(git rev-parse HEAD)
# echo checkout commit $commit

# srun -J lw -N 1 -p $cluster --gres gpu:$gpus \
CUDA_VISIBLE_DEVICES=0 
python3 train.py \
--en_wandb \
$exp_args ${@}