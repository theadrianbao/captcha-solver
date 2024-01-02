import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from opencv_process import process_captcha
import argparse
import os
from doctr.models import crnn_vgg16_bn

default_model = 'models/custom_captcha_crnn.pt'

"""
Instructions:
Using default CRNN model:
python inference.py path/to/your/image.png

Using your own custom-trained CRNN model with --reco_model_path:
python inference.py path/to/your/image.png --reco_model_path your_model.pt
"""

def setup_model(reco_model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False)
    model_dict = model.state_dict()

    if reco_model_path is not None and os.path.exists(reco_model_path):
        finetuned_rec_model = torch.load(reco_model_path, map_location=device)

        model_dict.update(finetuned_rec_model)

        model.load_state_dict(model_dict)
    else:
        print(f"Warning: No valid model loaded. Using the default model.")

    model.to(device)
    model.eval()

    return model


def process_image(image_path):
    image = process_captcha(image_path)

    pil_image = Image.fromarray(image).convert('RGB')

    preprocess = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
    ])

    input_img = preprocess(pil_image).unsqueeze(0)

    return input_img


def main():
    parser = argparse.ArgumentParser(description='Perform recognition on a captcha image')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--reco_model_path', type=str, default=f'{default_model}',
                        help=f'Path to the recognition model (default: {default_model}')
    args = parser.parse_args()

    model = setup_model(args.reco_model_path)
    input_tensor = process_image(args.image_path)
    output = model(input_tensor)

    print("Output:", output['preds'][0][0])


if __name__ == "__main__":
    main()