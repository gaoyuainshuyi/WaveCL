CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python3 main.py -c ./configs/refer_youtube_vos_finetune.yaml -rm train -ng 6 --epochs 25 \
-pw "runs/coco/coco_pretrain_tiny/checkpoints/27.pth.tar" --version "finetuneytb_tiny" \
--lr_drop 10 -bs 1 -ws 8 --backbone "video-swin-t" \
-bpp "pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth"

### ng means num_gpu -pw means the path of the pretrained weights lr_drop means the scheduler
