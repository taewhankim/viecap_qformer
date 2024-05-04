import torch
import json
import math
from PIL import Image
# from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
# from transformers import CLIPFeatureExtractor, CLIPVisionModel, CLIPProcessor, AutoProcessor
# device = "cuda" if torch.cuda.is_available() else "cpu"
# encoder_name = 'openai/clip-vit-base-patch32'
# feature_extractor = CLIPFeatureExtractor.from_pretrained(encoder_name)
# feature_extractor = AutoProcessor.from_pretrained(encoder_name)

# topil = ToPILImage()

def overlay_images(back_imgs, front_imgs, model, alpha=60):

    img_list = []
    for front_img, back_img in zip(front_imgs, back_imgs):
        front_img = topil(front_img)

        back_img = topil(back_img)
        # filename1 = back_img # jpg
        # front_img.save("/home/twkim/newf.png", format="png")
        # back_img.save("/home/twkim/newb.png", format="png")
        # Open Front Image
        # frontImage = Image.open(filename)
        frontImage = front_img.convert("RGBA")

        background = back_img.convert("RGBA")
        frontImage.putalpha(alpha) # 60
        # front_img.save("/home/twkim/newf_alpha.png", format="png")

        background.paste(frontImage, (0, 0), frontImage)

        # background.save("/home/twkim/new_result.png")
        img_list.append(background.convert("RGB"))
    with torch.no_grad():
        # pixel_values = feature_extractor(img_list, return_tensors='pt').pixel_values.to(device)
        # result = model(pixel_values=pixel_values).last_hidden_state
        inputs = feature_extractor(images=img_list, return_tensors="pt").to(device)
        result = model(**inputs).last_hidden_state
    return result


# Turns a dictionary into a class
class Dict2Class(object):

    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                                        sort_keys=True, indent=4)

def noise_injection(x, variance = 0.001, device = 'cuda') -> torch.Tensor:
    """
    Args:
        x: tensor with a shape of (batch_size, clip_hidden_size), prefix
        variance: the variance of noise
    Return:
        prefix with noise
    """
    if variance == 0.0:
        return x
    std = math.sqrt(variance)
    # normalization
    x = torch.nn.functional.normalize(x, dim = -1)
    # adding noise
    x = x + (torch.randn(x.shape, device = device) * std)

    return torch.nn.functional.normalize(x, dim = -1)