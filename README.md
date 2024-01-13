训练用train_diffusion.py
测试用eval_diffusion.py
计算指标使用calculate_psnr_ssim（可计算PSNR,SSIM,LPIPS）
环境：
GPU：2gpu(s)，每个GPU显存：24 GB
CPU：16core(s)，内存：48 GB
基于 PyTorch2.0.1 / Tensorflow2.13.1 / Paddle2.5.2 版本的docker镜像，horovod已集成，cuda11.8.0，conda3.10，ubuntu20.04

训练命令行：python train_diffusion.py --config /gemini/code/config/allweather.yml --resume /gemini/pretrain2/AllWeather_ddpm.pth.tar --train_dir $GEMINI_DATA_OUT
测试命令行：CUDA_VISIBLE_DEVICES=1 python eval_diffusion.py --config /gemini/code/config/allweather.yml --resume /gemini/pretrain2/20240112034941.pth.tar --test_set snow