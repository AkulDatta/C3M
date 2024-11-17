if command -v nvidia-smi &> /dev/null; then
    echo "CUDA GPU detected, running with GPU acceleration..."
    python main.py --log log_CAR --task CAR
else
    echo "No CUDA GPU detected, running with CPU only..."
    python main.py --log log_CAR --task CAR --no_cuda
fi

python plot.py --pretrained log_CAR/controller_best.pth.tar --task CAR --plot_type 3D --plot_dims 0 1 2
python plot.py --pretrained log_CAR/controller_best.pth.tar --task CAR --plot_type 2D --plot_dims 0 1
python plot.py --pretrained log_CAR/controller_best.pth.tar --task CAR --plot_type error