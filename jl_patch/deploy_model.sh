CUDA_VISIBLE_DEVICES=0 \
vllm serve /mnt/hdfs/tiktok_aiic/user/junlongli/models/Qwen/Qwen3-VL-8B-Instruct/ \
--port 21100 \
--served-model-name Qwen3-VL-8B-Instruct \
--tensor-parallel-size 1