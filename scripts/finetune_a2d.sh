CUDA_VISIBLE_DEVICES=2,3 python3 main.py -c ./configs/a2d_sentences_finetune.yaml -rm resume_train -ng 2 --epochs 20 \
-pw "runs/coco/coco_pretrain_tiny/checkpoints/29.pth.tar" --version "finetune_a2d" \
--lr_drop 20 -bs 2 -ws 8 --backbone "video-swin-t" \
-bpp "pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth"

#finetune a2d, NOTE the number gpu lr_drop
