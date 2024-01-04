import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, models
import argparse
import numpy as np
import json
import os
import random
from PIL import Image
from common import load_wights, load_cat_names


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--top_k', dest='top_k', default='5')
    parser.add_argument('--filepath', dest='filepath', default=None)
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', default=True)
    return parser.parse_args()


def image_process(image):
    n_size = [0, 0]

    if image.size[0] > image.size[1]:
        n_size = [image.size[0], 256]
    else:
        n_size = [256, image.size[1]]

    image.thumbnail(n_size, Image.ANTIALIAS)
    image_wight, image_height = image.size

    image_left = (256 - 224) / 2
    image_top = (256 - 224) / 2
    image_right = (256 + 224) / 2
    image_bottom = (256 + 224) / 2

    image = image.crop((image_left, image_top, image_right, image_bottom))

    image = np.array(image)
    image = image / 255.

    image_mean = np.array([0.485, 0.456, 0.406])
    image_std = np.array([0.229, 0.224, 0.225])
    image = (image - image_mean) / image_std

    image = np.transpose(image, (2, 0, 1))

    return image


def fn_predict(image_path, model, topk, gpu):
    model.eval()
    cuda = torch.cuda.is_available()
    if gpu and cuda:
        print("now using GPU")
        model = model.cuda()
    else:
        print('now using CPU')
        model = model.cpu()

    image_to_process = Image.open(image_path)
    xx_arr = image_process(image_to_process)
    tensor = torch.from_numpy(xx_arr)

    if gpu and cuda:
        enter = Variable(tensor.float().cuda())
    else:
        enter = Variable(tensor)

    enter = enter.unsqueeze(0)
    result = model.forward(enter)

    p_s = torch.exp(result).data.topk(topk)
    prob = p_s[0].cpu()
    predict_classes = p_s[1].cpu()
    class_p = {model.class_to_idx[k]: k for k in model.class_to_idx}
    cls_map = list()

    for l in predict_classes.numpy()[0]:
        cls_map.append(class_p[l])

    return prob.numpy()[0], cls_map


def main():
    m_arreg = parse_args()
    gpu = m_arreg.gpu
    model = load_wights(m_arreg.checkpoint)
    names_json_cat = load_cat_names(m_arreg.category_names)
    if m_arreg.filepath == None:
        number_of_image = random.randint(1, 102)
        m_image = random.choice(os.listdir('./flowers/test/' + str(number_of_image) + '/'))
        img_path = './flowers/test/' + str(number_of_image) + '/' + m_image
        train_probb, train_class = fn_predict(img_path, model, int(m_arreg.top_k), gpu)
        print('now image is select........ ' + str(names_json_cat[str(number_of_image)]))
    else:
        img_path = m_arreg.filepath
        train_probb, train_class = fn_predict(img_path, model, int(m_arreg.top_k), gpu)
        print('now file is select...... ' + img_path)
    print('the probability is {} : '.format(train_probb))
    print('the classes {}'.format(train_class))
    print("the json file: ")
    print([names_json_cat[w] for w in train_class])


if __name__ == "__main__":
    main()