CUDA_VISIBLE_DEVICES=2,3,4,5 python3 main_pretrain.py -c ./configs/coco.yaml -rm train --version "coco_pretrain_tiny" -ng 4 --epochs 30 \
--lr_drop 15 20 -bs 6 --backbone "video-swin-t" \
-bpp "pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth"
