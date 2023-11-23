import io
import torch.onnx
from model import U2NET
import os
from model_processing.prepare_model import  get_latest_version

def convert_model_to_onnx(model_path):
    torch_model = U2NET(3,1)
    batch_size = 1

    torch_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    torch_model.eval()

    x = torch.randn(batch_size, 3, 512, 512, requires_grad=True)
    last_character = get_latest_version(model_path)
    model_dir = "saved_models/ABR_model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.onnx.export(torch_model, x,os.path.join(model_dir,f"ARB_version_{int(last_character)}.onnx"), 
                      export_params=True, 
                      opset_version=11, 
                      do_constant_folding=True, 
                      input_names = ['input'], 
                      output_names = ['output'], 
                      dynamic_axes = {'input' : {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                      )


if __name__ == '__main__':
    pass