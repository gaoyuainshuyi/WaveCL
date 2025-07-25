CUDA_VISIBLE_DEVICES=0,1 python3 infer_refytb.py -c ./configs/refer_youtube_vos_finetune.yaml -rm test --version "scratch_tiny_test" -ng 2 --backbone "video-swin-t" \
-bpp "pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth" \
-ckpt "runs/ref_youtube_vos/finetuneytb_tiny/checkpoints/18.pth.tar"

