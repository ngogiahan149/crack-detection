import streamlit as st
from utils import load_unet_resnet_101, load_unet_resnet_34, load_unet_vgg16
import gc
import tqdm 
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from os.path import join
from pathlib import Path
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch
import cv2 as cv
from unet.unet_transfer import UNet16, input_size
def main():
    st.title('Crack detection')
    model_option = ['None','vgg 16', 'resnet 34', 'resnet 101']
    select_model = st.radio("Select model below", model_option)

    #Define model path
    vgg16_path = r'D:\NCKU\Intelligent Manufacturing System\Final project\streamlit\models\model_unet_vgg_16_best.pt'
    resnet34_path = r'D:\NCKU\Intelligent Manufacturing System\Final project\streamlit\models\model_unet_resnet34_best.pt'
    resnet101_path = r'D:\NCKU\Intelligent Manufacturing System\Final project\streamlit\models\model_unet_resnet101_best.pt'

    if select_model == 'None':
        st.write('None is chosen')
    elif select_model == 'vgg 16':
        st.write('Model chosen: vgg 16')
        model = load_unet_vgg16(vgg16_path)
        detect_crack(model)
    elif select_model == 'resnet 34':
        st.write('Model chosen: resnet 34')
        model = load_unet_resnet_34(resnet34_path)
        detect_crack(model)
    else:
        st.write('Model chosen: resnet 101')
        model = load_unet_resnet_101(resnet101_path)
        detect_crack(model)
def detect_crack(model1):
    paths = [path for path in Path(img_dir).glob('*.*')]
    print(paths[1])
    for path in paths:
        print(str(path))

        # Open input image
        
        img_0 = Image.open(str(path)).convert('RGB')
        img_0 = np.asarray(img_0)
        print(img_0.shape)
        img_0 = img_0[:,:,:3]
        img_height, img_width, img_channels = img_0.shape
        if len(img_0.shape) != 3:
            print(f'incorrect image shape: {path.name}{img_0.shape}')
            continue
        


            
        if img_0.shape[0] > 1000 or img_0.shape[1] > 1000:
            img_1 = cv.resize(img_0, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
        else:
            img_1 = img_0
            # plt.subplot(122)
            # plt.imshow(img_0), plt.title(f'{img_0.shape}')
            # plt.show()
        img_height, img_width, img_channels = img_1.shape
        prob_map_patch = evaluate_img(model1, img_1)
        output = prob_map_patch
        output = torch.sigmoid(prob_map_patch[0, 0]).data.cpu().numpy()
        output = cv.resize(output, (img_width, img_height), cv.INTER_AREA)

        fig = plt.figure()
        title = fig.suptitle(f'name={path.stem} \n cut-off threshold = 0.2', fontsize="medium")
        ax = fig.add_subplot(131)
        ax.imshow(img_1)
        ax = fig.add_subplot(132)
        ax.imshow(output)
        ax = fig.add_subplot(133)
        ax.imshow(img_1)
        ax.imshow(output, alpha=0.4)

        st.pyplot(fig)
        plt.close('all')
    gc.collect()

def evaluate_img(model, img):
    input_width, input_height = input_size[0], input_size[1]

    img_1 = cv.resize(img, (input_width, input_height), cv.INTER_AREA)
    X = train_tfms(Image.fromarray(img_1))
    X = Variable(X.unsqueeze(0))  # [N, 1, H, W]

    mask = model(X)
    print("Predicted: ",mask.shape)
    # criterion = nn.BCEWithLogitsLoss().to('cpu')
    # criterion(mask, output_target)
    # # mask = torch.sigmoid(mask[0, 0]).data.cpu().numpy()
    # mask = cv.resize(mask, (img_width, img_height), cv.INTER_AREA)
    return mask

def evaluate_img_patch(model, img):
    input_width, input_height = input_size[0], input_size[1]

    img_height, img_width, img_channels = img.shape

    if img_width < input_width or img_height < input_height:
        return evaluate_img(model, img)

    stride_ratio = 0.1
    stride = int(input_width * stride_ratio)

    normalization_map = np.zeros((img_height, img_width), dtype=np.int16)

    patches = []
    patch_locs = []
    for y in range(0, img_height - input_height + 1, stride):
        for x in range(0, img_width - input_width + 1, stride):
            segment = img[y:y + input_height, x:x + input_width]
            normalization_map[y:y + input_height, x:x + input_width] += 1
            patches.append(segment)
            patch_locs.append((x, y))

    patches = np.array(patches)
    if len(patch_locs) <= 0:
        return None

    preds = []
    for i, patch in enumerate(patches):
        patch_n = train_tfms(Image.fromarray(patch))
        X = Variable(patch_n.unsqueeze(0)) # [N, 1, H, W]
        masks_pred = model(X)
        mask = torch.sigmoid(masks_pred[0, 0]).data.cpu().numpy()
        preds.append(mask)

    probability_map = np.zeros((img_height, img_width), dtype=float)
    for i, response in enumerate(preds):
        coords = patch_locs[i]
        probability_map[coords[1]:coords[1] + input_height, coords[0]:coords[0] + input_width] += response

    return probability_map

def disable_axis():
    plt.axis('off')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_ticklabels([])
    plt.gca().axes.get_yaxis().set_ticklabels([])
if __name__ == "__main__":
    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])
    img_dir = r'D:\NCKU\Intelligent Manufacturing System\Final project\crack_segmentation_dataset\test\images'
    main()