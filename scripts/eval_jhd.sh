CUDA_VISIBLE_DEVICES=0,1 python3 main.py -c ./configs/jhmdb_sentences.yaml -rm test -ng 2 --version "jhd_test" --backbone "video-swin-t" \
-bpp "pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth" \
-bs 2
## test mode the version can be ignored but necessary to be given
