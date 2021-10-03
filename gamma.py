
import cv2 as cv
import sys
import numpy as np
import torch
import rawpy
import os
from collections import OrderedDict

sys.path.insert(0,os.path.abspath('../../codes/models/archs'))
sys.path.insert(0,os.path.abspath('../../codes/models/'))
sys.path.insert(0,os.path.abspath('../../codes'))
sys.path.insert(0,os.path.abspath('../../'))
import codes
from codes.models.archs.CEL import CEL_net
import codes.utils.util as util
exposure_max = 200
alpha_max = 100
title_window = 'gamma corection'
global img_dark 
global netG
global alpha_g
global exp_g

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_network(load_path, network, strict=True):
       # print(os.getcwd())
       # if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
       #     network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)

def pack_raw(raw):
    #pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32) 
    im = np.maximum(im - 512,0)/ (16383 - 512) #subtract the black level

    im = np.expand_dims(im,axis=2) 
    img_shape = im.shape
    
    H = img_shape[0] - img_shape[0]%32
    W = img_shape[1] - img_shape[1]%32
    out = np.concatenate((im[0:H:2,0:W:2,:], 
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:]), axis=2)
    return out

def on_exposure(val):
    global netG
    global img_dark
    global alpha_g
    global exp_g
    exp_g =  torch.tensor(val).to(device)
    im = netG(img_dark , exp_g , alpha_g)
    im = util.tensor2img(im)
    cv.imshow(title_window,im )
    #cv.imshow(title_window,cv.resize(im, (1560, 512)) )

def on_alpha(val):
    global netG
    global img_dark
    global alpha_g
    global exp_g
    alpha_g =  float(val)/100
    alpha_g = torch.tensor(alpha_g).to(device)
    im = netG(img_dark , exp_g , alpha_g)
    im = util.tensor2img(im)
    cv.imshow(title_window,im )
    #cv.imshow(title_window,cv.resize(im, (1560, 512)) )

def Display():
    
    cv.namedWindow(title_window)
    trackbar_name = 'exposure x %d' % exposure_max
    cv.createTrackbar(trackbar_name, title_window , 10, exposure_max, on_exposure)
    trackbar_name = 'alpha '
    cv.createTrackbar(trackbar_name, title_window , 0, alpha_max, on_alpha)
    # Show some stuff
    on_exposure(torch.tensor(10).to(device))
    
    # Wait until user press some key
    while True:
        k = cv.waitKey(100) # change the value from the original 0 (wait forever) to something appropriate
        if k == 27:
            print('ESC')
            cv.destroyAllWindows()
            break        
        if cv.getWindowProperty(title_window,cv.WND_PROP_VISIBLE) < 1:        
            break        
    cv.destroyAllWindows()

def main():
    global netG
    global img_dark
    global alpha_g
    global exp_g
    alpha_g = torch.tensor(0).to(device)
    exp_g = torch.tensor(100).to(device)
    ps_x = 768
    ps_y = 384
    yy = 500
    xx = 1000
   
    im_path = sys.argv[1]
    raw = rawpy.imread(im_path)
    img_dark = pack_raw(raw)
    print(img_dark.shape)
    img_dark = img_dark[yy:yy+ps_y,xx:xx+ps_x,:]
    img_dark = np.maximum(np.minimum(img_dark,1.0),0.0)
    img_dark = torch.from_numpy(img_dark).permute(2,0,1)
    img_dark = torch.unsqueeze(img_dark, 0)
    img_dark = img_dark.to(device)
    print(img_dark.get_device())
    netG = CEL_net(4, 3) 
    load_network("../../experiments/CEL_lowp_tune_office/models/latest_G.pth",netG)
    netG.to(device)
    netG.eval()

    Display()


if __name__ == '__main__':
    main()
