import onnx
import torch
from model import U2NET

TORCH_PATH = "./saved_models/u2net_portrait/u2net_portrait.pth"
ONNX_PATH = "/Users/ttjiaa/Pictures/Code/ml/converted/my_model.onnx"
input_size = (1, 3, 224, 224)

# load pytorch model
net = U2NET(3,1)
net.load_state_dict(torch.load(TORCH_PATH, map_location=torch.device('cpu')))
if torch.cuda.is_available():
    net.cuda()
net.eval()

torch.onnx.export(
    model=net,
    args=torch.randn(*input_size), # input image size, follow with (N,H,W,C) format
    f=ONNX_PATH, # where should it be saved
    verbose=False,
    export_params=True,
    do_constant_folding=False,  # fold constant values for optimization
    # do_constant_folding=True,   # fold constant values for optimization
    input_names=['input'],
    output_names=['output']
)
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)