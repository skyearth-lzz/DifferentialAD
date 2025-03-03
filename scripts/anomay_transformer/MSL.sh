# 这行代码用于指定使用第8号GPU (从0开始计数,所以7表示第8块GPU)
# 通过设置CUDA_VISIBLE_DEVICES环境变量来控制程序只能看到和使用指定的GPU设备
export CUDA_VISIBLE_DEVICES=0

# 用于训练模型,各参数含义如下:
# --anormly_ratio 1: 异常比例为1
# --num_epochs 3: 训练3个epoch
# --batch_size 256: 每个batch包含256个样本
# --mode train: 训练模式
# --dataset MSL: 使用MSL数据集
# --data_path dataset/MSL: 数据集路径
# --input_c 55: 输入通道数为55（时间序列的维度）
# --output_c 55: 输出通道数为55（时间序列的维度）
python main.py --anormly_ratio 1 --num_epochs 10   --batch_size 256  --mode train --dataset MSL  --data_path dataset/MSL --input_c 55    --output_c 55

# 用于测试模型,各参数含义如下:
# --anormly_ratio 1: 异常比例为1  
# --num_epochs 10: 测试10个epoch
# --batch_size 256: 每个batch包含256个样本
# --mode test: 测试模式
# --dataset MSL: 使用MSL数据集
# --data_path dataset/MSL: 数据集路径
# --input_c 55: 输入通道数为55
# --output_c 55: 输出通道数为55
# --pretrained_model 20: 使用第20个epoch保存的预训练模型
python main.py --anormly_ratio 1  --num_epochs 10      --batch_size 256     --mode test    --dataset MSL   --data_path dataset/MSL  --input_c 55    --output_c 55  --pretrained_model 20




