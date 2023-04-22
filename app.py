import PIL.Image as Image
import numpy as np
import matplotlib
import requests
matplotlib.use('agg') 
from matplotlib import pyplot as plt
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch
from io import BytesIO
import gradio as gr
from torchvision import transforms


transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

# device = torch.device("cpu")
# model = CSRNet()
# model = model.to(device)
# checkpoint = torch.load('partBmodel_best.pth.tar', map_location='cpu')
# model.load_state_dict(checkpoint['state_dict'])

if torch.cuda.is_available():
    print("cuda is available, original weights")
    device = torch.device("cuda")
    model = CSRNet()
    model.to(device)
    checkpoint = torch.load('partBmodel_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
else:
    print("cuda is not available, cpu weights")
    device = torch.device("cpu")
    model = CSRNet()
    model.to(device)
    checkpoint = torch.load('partBmodel_best.pth.tar', map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

model.eval()
def people(img_pil):
    img = transform(img_pil.convert('RGB')).to(device)
    output = model(img.unsqueeze(0))
    output_density = np.squeeze(output.detach().cpu().numpy())
    plt.imshow(output_density,cmap=CM.jet)
    tmpfile = BytesIO()
    plt.savefig(tmpfile)
    img_pil_density = Image.open(tmpfile)
    people_number = np.around(output.detach().cpu().sum().numpy())
    plt.clf()
    return img_pil_density, people_number

with gr.Blocks() as demo:
    gr.Markdown("""
        # people count
    """)
    with gr.Row():
        with gr.Column():
            inp = gr.Image(source="webcam", type='pil', streaming=True)
        with gr.Column():
            out = gr.Image(label='Out-Calibration')
            peo_count = gr.Number(label="people number")
        inp.change(people, inp, [out,peo_count])

if __name__ == "__main__":
    # demo.launch(server_name="0.0.0.0", server_port = 6006, share = True)
    demo.launch()