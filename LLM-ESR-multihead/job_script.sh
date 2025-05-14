#!/bin/bash
#SBATCH --job-name=beauty_exp         # 作业名称
#SBATCH --output=slurm-%j.out         # 标准输出文件
#SBATCH --error=slurm-%j.err          # 错误输出文件
#SBATCH --mail-user=nl2752@nyu.edu    # 邮箱地址
#SBATCH --mail-type=ALL               # 邮件通知类型
#SBATCH --ntasks=1                    # 只需要 1 个任务
#SBATCH --cpus-per-task=1             # 每个任务 1 个 CPU
#SBATCH --gres=gpu:1                  # 请求 1 个 GPU
#SBATCH --mem=64G                     # 请求 64 GB 内存
#SBATCH --time=2:00:00                # 运行时间限制为 2 小时

# 如果需要加载模块（根据集群环境）
# module load cuda/11.7

# 激活 conda 环境
source /scratch/nl2752/ml/bin/activate

# 运行 bash 脚本
bash experiments/beauty.bash