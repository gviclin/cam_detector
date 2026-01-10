Some test to learn about deep learning 

python -c "import onnx; model = onnx.load('yolov8n.onnx'); onnx.checker.check_model(model)"

wget https://github.com/bluenviron/mediamtx/releases/latest/download/mediamtx_linux_amd64.tar.gz
tar -xzf mediamtx_linux_amd64.tar.gz

./mediamtx

rtsp://192.168.68.64:8554/