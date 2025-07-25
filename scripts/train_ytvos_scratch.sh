python3 main.py -c ./configs/refer_youtube_vos.yaml -rm train -ng 6 --epochs 30 \
--version "ytb_from_scratch" --lr_drop 20 -bs 1 -ws 8 --backbone "video-swin-t" \
-bpp "pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth"


