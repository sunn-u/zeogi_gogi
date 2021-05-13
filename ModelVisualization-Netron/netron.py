import torch
import torchvision # model을 불러오기 위해 import 하였습니다.
import torch.onnx

from vggModule import VGG
import netron


# 1. 임의의 model을 사용해도 되며, 실제 사용하는 custom model을 불러와서 저장해 보시기 바랍니다.
model = model = VGG(conv, num_classes=7, init_weights=True)

# 2. model의 파라미터를 OrderedDict 형태로 저장합니다.
params = model.state_dict()

# 3. 동적 그래프 형태의 pytorch model을 위하여 data를 model로 흘려주기 위한 더미 데이터 입니다.
dummy_data = torch.empty(1, 1, 224, 224, dtype = torch.float32)
#dummy_data = torch.empty(1, 3, 224, 224, dtype = torch.float32)

# 4. onnx 파일을 export 해줍니다. 함수에는 차례대로 model, data, 저장할 파일명 순서대로 들어가면 됩니다.
torch.onnx.export(model, dummy_data, "MyVGG16.onnx")


# !pip install netron
netron.start('E:\\Python_Project\\vgg16.onnx')