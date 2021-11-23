import tqdm
import numpy as np
import torch
import torchvision.models as m
def cls_predict(model, imgs_list):
    print('..... Finished loading model! ......')
    imgs_list = np.stack(imgs_list, 0)
    imgs_list = torch.from_numpy(imgs_list).float()
    imgs_list /= 255.0
    imgs_list -= torch.Tensor([0.485, 0.406, 0.456]) 
    imgs_list /= torch.Tensor([0.229, 0.224, 0.225])
    imgs_list = imgs_list.permute(0, 3, 1, 2)
    if torch.cuda.is_available():
        model = model.cuda()
        imgs_list = imgs_list.cuda()
    with torch.no_grad():
        out = model(imgs_list)
    predictions = torch.argmax(out, dim=1).cpu()
    return predictions

def init_classifier(weight):
    model = m.mobilenet_v2()
    model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=2, bias=True)
    model.load_state_dict(torch.load(weight))
    model.eval()
    return model
    
if __name__ == "__main__":
    model = _init_classifier("")
