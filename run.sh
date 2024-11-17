if [ $# -eq 0 ]; then
    echo "Please provide a task name as argument"
    echo "Usage: $0 <task_name>"
    exit 1
fi

TASK=$1

if command -v nvidia-smi &> /dev/null; then
    echo "CUDA GPU detected, running with GPU acceleration..."
    python main.py --log log_${TASK} --task ${TASK}
else
    echo "No CUDA GPU detected, running with CPU only..."
    python main.py --log log_${TASK} --task ${TASK} --no_cuda
fi

python plot.py --pretrained log_${TASK}/controller_best.pth.tar --task ${TASK} --plot_type 3D --plot_dims 0 1 2
python plot.py --pretrained log_${TASK}/controller_best.pth.tar --task ${TASK} --plot_type 2D --plot_dims 0 1
python plot.py --pretrained log_${TASK}/controller_best.pth.tar --task ${TASK} --plot_type error