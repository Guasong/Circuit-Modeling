#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# Define prediction lengths
pred_lens=(24 48 168 336 720)
model="Contiformer" # Change this to "Autoformer" to run the Autoformer model

# Loop through prediction lengths
for pred_len in "${pred_lens[@]}"; do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_$pred_len \
    --model $model \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 \
    --learning_rate 0.01
done
