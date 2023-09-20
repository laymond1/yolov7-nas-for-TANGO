TARGET=galaxy_s22_nnapi
PRED_TARGET=head

CUDA_VISIBLE_DEVICES='2,3' \
python lat_pred/lat_pred_train.py\
    --target $TARGET\
    --pred_target $PRED_TARGET\
    --epochs 1000\
    --batch_size 500\
    --lr 1e-4\
    --use_wandb
