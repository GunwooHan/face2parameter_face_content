#CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --gpus 4 --batch_size 16 --epochs 500 --model_pretrained True --name resnet50_torchvision_pretrained
#CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --gpus 4 --batch_size 16 --epochs 500 --model_pretrained False --name resnet50_torchvision

CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --gpus 4 --batch_size 64 --epochs 500 --learning_rate 0.001 --model_pretrained True --name resnet50_torchvision_pretrained
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --gpus 4 --batch_size 64 --epochs 500 --learning_rate 0.001 --model_pretrained False --name resnet50_torchvision
