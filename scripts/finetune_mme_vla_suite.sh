# A total of 14 VLA variants are considered in our experiments:
#  FrameSamp                        TokenDrop                       RMT                       TTT                      Symbolic
# perceptual-framesamp-context  perceptual-tokendrop-context  recurrent-rmt-context  recurrent-ttt-context  symbolic-grounded-subgoal
# perceptual-framesamp-expert   perceptual-tokendrop-expert   recurrent-rmt-expert   recurrent-ttt-expert   symbolic-simple-subgoal
# perceptual-framesamp-modul    perceptual-tokendrop-modul    recurrent-rmt-modul    recurrent-ttt-modul

MME_VLA_TYPE="perceptual-framesamp-modul"

export WANDB_API_KEY=<YOUR_WANDB_API_KEY>

CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 uv run scripts/train.py mme_vla_suite \
--exp-name=${MME_VLA_TYPE}_your_model_name \
--batch-size=64 \
--num-workers=4 \
--fsdp-devices=4 \
--dataset-path=data/robomme_preprocessed_data \
--dataset-type=bin --num-read-threads=1 \
--model.use_history \
--model.history_config="${MME_VLA_TYPE}.yaml"

# --dataset-type=bin (instead of the "npy" default): reads history features from
# the single-file-per-episode format produced by scripts/convert_features_to_bin.py
# via seek/pread instead of opening ~32 separate small .npy files per sample --
# measured 9x+ dataloader throughput on this config (see tests/speed_testing.py).
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.85, not 0.95: at 0.95 JAX preallocates so much
# of each A100 40GB that NCCL has no headroom for its own collective-op buffers,
# and the first FSDP all-gather crashes with a deterministic
# "ncclGroupEnd() failed ... out of memory".