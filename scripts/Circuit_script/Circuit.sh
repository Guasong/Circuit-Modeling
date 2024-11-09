#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# Define prediction lengths
pred_lens=(501)
model="Contiformer" # Change this to "Autoformer" to run the Autoformer model

# Loop through prediction lengths
for pred_len in "${pred_lens[@]}"; do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/adc/ \
    --data_path adc_small.csv \
    --model_id adc_small_$pred_len \
    --model $model \
    --data Circuit \
    --seq_len $pred_len \
    --label_len 0 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 3 \
    --dec_in 2 \
    --c_out 2 \
    --des 'Exp' \
    --itr 1
done
