import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os 
class Img2Vec():

    def __init__(self, cuda=False):
        """ Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        """
        self.device = torch.device("cuda" if cuda else "cpu")
        self.model = models.resnet50(pretrained=True).to(self.device)
        self.extraction_layer = self.model._modules.get('avgpool')
        self.layer_output_size = 2048

        self.model.eval()

        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def get_vec(self, img, tensor=False):
        """ Get vector embedding from PIL image
        :param img: PIL Image or list of PIL Images
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
        if type(img) == list:
            a = [self.normalize(self.to_tensor(self.scaler(im))) for im in img]
            images = torch.stack(a).to(self.device)
            
            my_embedding = torch.zeros(len(img), self.layer_output_size, 1, 1)

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            h_x = self.model(images)
            h.remove()

            if tensor:
                return my_embedding
            else:
                return my_embedding.numpy()[:, :, 0, 0]
        else:
            image = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0).to(self.device)

            my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            h_x = self.model(image)
            h.remove()

            if tensor:
                return my_embedding
            else:
                return my_embedding.numpy()[0, :, 0, 0]


img2vec = Img2Vec(True)
#change to train/test/valid
directory = "../../../media/hdd/fodor/db/fi_processed/train/"
i = 0
for videos in os.listdir(directory):
    imgs = []
    for filename in os.listdir(os.path.join(directory, videos, "frames")):
        imgs.append(img2vec.get_vec(Image.open(os.path.join(directory, videos, "frames", filename))))
    i+=1
    np_img=np.asarray(imgs)
    #change to train/test/valid
    np.save(os.path.join("db/video/train", videos), np_img)
    print("img " + str(i))