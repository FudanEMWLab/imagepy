# -*- coding: utf-8 -*-

from imagepy.core.engine import Filter
from imagepy.core.hough import hough
from imagepy import IPy
import numpy as np

class Hough(Filter):
    title = 'Hough'
    note = ['all', 'auto_msk', 'auto_snap', 'not_channel']
  
    def run(self, ips, snap, img, para = None):
        print(snap.shape)
        img = hough.crop_one_circle(snap)
        img = np.expand_dims(img, axis=0)
        IPy.show_img(img, 'img')

plgs = [Hough]