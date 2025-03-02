export CUDA_VISIBLE_DEVICES=1

python main.py --anormly_ratio 0.5 --num_epochs 10 --n_heads 8 --d_model 256 --layers 6  --batch_size 256  --mode train --dataset SMD  --data_path dataset/SMD   --input_c 38
python main.py --anormly_ratio 0.5 --num_epochs 10 --n_heads 8 --d_model 256 --layers 6  --batch_size 256     --mode test    --dataset SMD   --data_path dataset/SMD     --input_c 38     --pretrained_model 20