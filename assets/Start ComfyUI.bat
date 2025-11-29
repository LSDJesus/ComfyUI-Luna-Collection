@echo off

cd /d "%~dp0"

call .\venv\scripts\activate

:: set NUMEXPR_MAX_THREADS=16
set CUDA_VISIBLE_DEVICES=0,1

python.exe main.py --listen --port 8188 --cuda-malloc --fast fp16_accumulation --normalvram --use-sage-attention

:: --use-flash-attention --use-sage-attention --cache-lru 10

pause