import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, psnr
from model import srcnn_model
import streamlit as st


st.set_page_config(page_title="SR X-RAY SCAN", page_icon=None, layout='centered', initial_sidebar_state='auto')


html_temp = """
<div style ="background-color:#B6F2E2;padding:13px">
<h1 style ="color:black;text-align:center;">Super Resolution of Medical X-Ray Scans</h1>
</div>
"""

st.markdown(html_temp, unsafe_allow_html = True)

html_temp = """
<br><br>
"""

st.markdown(html_temp, unsafe_allow_html = True)


weight_path = "weights/best.pth"
scale = 2

cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = srcnn_model().to(device)

state_dict = model.state_dict()
for n, p in torch.load(weight_path, map_location=lambda storage, loc: storage).items():
    if n in state_dict.keys():
        state_dict[n].copy_(p)
    else:
        raise KeyError(n)


model.eval()


image_file = st.file_uploader("Upload the X-Ray Scan", type = ["jpg", "png", "jpeg"])


if st.button("Show the uploaded Scan"):
    image = pil.open(image_file).convert('RGB')
    st.image(image, caption = "Uploaded X-Ray Scan.", use_column_width = "auto")
    st.success("Image Uploaded Successfully")



if st.button("Show the Super Resolution Scan"):
    image = pil.open(image_file).convert('RGB')
    image_width = (image.width ) * scale
    image_height = (image.height) * scale

    image = image.resize((image_width, image_height), resample=pil.BICUBIC)

    image = np.array(image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image)

    y = ycbcr[..., 0]
    y /= 255.

    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)


    avg_psnr = psnr(y, preds)

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil.fromarray(output)


    st.image(output, caption = "Super Resolution X-Ray Scan.", use_column_width = True)

    st.success("SR Image Genereated Successfully")
