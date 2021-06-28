# IMPORTS

from PIL import Image
import numpy as np
import torch
from torchvision import transforms


# FUNCTIONS
# Func 1: load and reshape image, convert it to pytorch tensor

def load_image(img_path, max_size=600, shape=(1, 1), get_shape=False):
    '''
    param:: img_path - path to image
    param:: max_size - reshape size
    param:: shape - side ratio
    param:: get_shape - helps save content, style and result images side ratio
    return:: batch with one image
    '''

    image = Image.open(img_path).convert('RGB')

    if get_shape:
        w, h = image.size
        over = w / h
        if over > 1:
            shape = (over, 1)
        elif over < 1:
            shape = (1, over)
        else:
            shape = (1, 1)

    in_transform = transforms.Compose([
        transforms.Resize((int(shape[0] * max_size), int(shape[1] * max_size))),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    image = in_transform(image)[:3, :, :].unsqueeze(0)

    if get_shape:
        return image, shape
    else:
        return image


# Func 2: just convert pytorch tensor to image

def im_convert(tensor):
    '''
    param:: tensor - tensor we need to convert
    return:: converted image with RGB[0;1]
    '''

    image = tensor.to("cpu").clone().detach()

    image = image.numpy().squeeze()

    image = image.transpose(1, 2, 0)

    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))

    image = image.clip(0, 1)

    return image


# Func 3: get dict of feature maps after vgg19 layers

def get_features(image, model, id):
    '''
    param:: image - image, that features we need
    param:: model - btw vgg
    param:: id - image name
    return:: dict of feature maps after different layers
    '''

    if id == 'content':
        layers = {'21': 'conv4_2'}
    elif id == 'style':
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '28': 'conv5_1'}
    elif id == 'target':
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',  # content layer
                  '28': 'conv5_1'}
    else:
        raise Exception

    features = {}
    x = image
    for name, layer in enumerate(model.features):
        x = layer(x)
        if str(name) in layers:
            features[layers[str(name)]] = x

    return features


# Func 4: get gram matrix

def gram_matrix(tensor):
    '''
    param:: tensor (1, *, *, *)
    return:: tensor gram matrix
    '''

    _, n_filters, h, w = tensor.size()

    tensor = tensor.view(n_filters, h * w)

    gram = torch.mm(tensor, tensor.t())

    return gram
