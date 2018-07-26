import logging
import sys
from random import shuffle

import cv2
import numpy as np
import torch
import torchvision.transforms as tforms
from colorama import Fore, Style
from torch.utils.data.dataset import Dataset

from data.utils.make_query import make_query, groupbyimages


class MongoAnn(Dataset):
    def __init__(self, mongoquery, classmap, shuffle_data=True,
                 bgr_format=False, random_flip=True, random_color=True,
                 random_seed=0):
        """
        Constructor
        :param mongoquery: A string representing a mongodb query to be made
        :param classmap: A string representing a dictionary with classname as
        keys and corresponding label as values
        :param shuffle_data: If True, shuffles the list of images (alongwith
        their annotations).
        :param bgr_format: If True, images are returned in BGR Format
        :param random_flip: If True, randomly do horizontal flipping
        :param random_color: If True, randomly apply color transform.
        :param random_seed: Seed for numpy.random
        """
        np.random.seed(random_seed)
        self._mongoquery = mongoquery
        self._random_flip = random_flip
        self._random_color = random_color
        self._colortform = tforms.Compose([tforms.ToPILImage(),
                                           tforms.ColorJitter(),
                                           tforms.ToTensor()])
        self._nocolortform = tforms.Compose([tforms.ToTensor()])
        self._bgr_format = bgr_format
        self._class2label = classmap
        # label2class is the reverse of the class2label dictionary
        print(self.class2label)
        self._label2class = {v: k for k, v in self.class2label.items()}
        # The format of mongoresults is as follows
        # [result1, result2,...., resultN]
        # Each result is a dictionary with the following format
        # {'filename':<Image file full path>, 'object': A dictionary}
        # The object dictionary is as follows
        # {'bbox' : A dictionary}
        # The bbox dictionary is as follows
        # {'xmin' : [List of xminx of bbox], 'ymin' : <Obvious>, 'xmax' :
        # <Obvious>, 'ymax' : <Obvious>,'text' : [List of class names],
        # 'label' : [List of label of the classes according to
        # self._class2label]
        self._mongoresults = groupbyimages(make_query(self.mongoquery),
                                           classmap)
        if shuffle_data:
            shuffle(self._mongoresults)

        if self.bgr_format:
            logging.WARNING(Fore.CYAN + Style.BRIGHT + 'Images will be '
                                                       'returned in BGR '
                                                       'format. It may affect '
                                                       'color'
                                                       ' augmentation results.')

    def __len__(self):
        return len(self.mongoresults)

    def __getitem__(self, index):
        """
        Returns an image and its annotations
        :param index: index of the data
        :return: A tuple (image, bboxes, labels). All elements
        of the tuple are TorchTensors. bboxes is of shape (num, 4). Its
        format is (xmin, ymin, xmax, ymax). All coordinates are unnormalized.
        """
        filename = self.mongoresults[index]['filename']
        try:
            image = cv2.imread(filename)
        except FileNotFoundError:
            logging.FATAL(Fore.RED + Style.BRIGHT + ' {} was not '
                                                    'found.'.format(filename))
            sys.exit(1)

        image_shape = image.shape

        if len(image.shape) == 2:
            logging.FATAL(Fore.RED + Style.BRIGHT + '{} is grayscale. '
                                                    'Only COLOR images '
                                                    'are '
                                                    'supported.'.format(
                filename))
        if not self.bgr_format:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gtinfo = self.mongoresults[index]
        labels = gtinfo['object']['bbox']['label']
        xmin = gtinfo['object']['bbox']['xmin']
        ymin = gtinfo['object']['bbox']['ymin']
        xmax = gtinfo['object']['bbox']['xmax']
        ymax = gtinfo['object']['bbox']['ymax']
        bboxes = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
        doflip = bool(np.random.randint(2))
        if doflip:
            image = cv2.flip(image, flipCode=1)
            # Next two lines flip the bounding box x coordinates.
            # Since this is horizontal flipping, y coordinates need no
            # transformation.
            bboxes[0, :] = image_shape[1] - bboxes[0, :] - 1
            bboxes[2, :] = image_shape[1] - bboxes[2, :] - 1

        docolor = bool(np.random.randint(2))
        if docolor:
            image = self.colortform(image)
        else:
            image = self.nocolortform(image)


        labels = torch.from_numpy(np.array(labels))
        bboxes = torch.from_numpy(bboxes)
        return image, bboxes, labels

    @property
    def mongoquery(self):
        return self._mongoquery

    @property
    def random_flip(self):
        return self._random_flip

    @property
    def random_color(self):
        return self._random_color

    @property
    def mongoresults(self):
        return self._mongoresults

    @property
    def class2label(self):
        return self._class2label

    @property
    def label2class(self):
        return self._label2class

    @property
    def bgr_format(self):
        return self._bgr_format

    @property
    def colortform(self):
        return self._colortform

    @property
    def nocolortform(self):
        return self._nocolortform
