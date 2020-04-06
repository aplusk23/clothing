import os
from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import numpy as np
from PIL import Image
from joblib import load
import requests
import re

clf = load('path/to/clf/filename.pkl') 
config_file = 'path/to/config/htc_dconv_c3-c5_mstrain_x101_64x4d_fpn_20e_1200x1900.py'
checkpoint_file = 'path/to/checkpoint.pth'

images =[]
names = []
import json 
import re
with open('/content/befree.json') as data:
  feed = json.load(data)
  for i in feed:
    images.append(i['image_url'])
    names.append(i['name'])
with open('/content/soeasy.json') as data:
  feed = json.load(data)
  for i in feed: 
    images.append(i['image_url'])
    names.append(i['name'])
names = names[::10]
images = images[::10]

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

model.CLASSES = ['shirt', 'top', 'sweater', 'cardigan',\
                   'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', \
                   'jumpsuit', 'cape', 'glasses', 'hat', 'headband, head covering, hair accessory', \
                   'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights, stockings', \
                   'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar',\
                   'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle',\
                   'zipper', 'applique', 'bead', 'bow', 'flower', 'fringe', 'ribbon',\
                   'rivet', 'ruffle', 'sequin', 'tassel']


#classes for main attr prediction
clss = [['shirt', 'top', 'sweater', 'cardigan', 'dress'],\
        ['shorts', 'skirt', 'pants', 'jumpsuit', 'leg warmer', 'tights, stockings'], \
        ['shoe'], ['bag, wallet'], ['cape', 'glasses', 'hat', 'headband, head covering, hair accessory','tie',\
                                    'glove', 'watch', 'belt', 'sock', 'scarf', 'umbrella'],['jacket','vest','coat']]                   

              

c=0
for image_url,nm in zip(images,names):
  c+=1
  if c%10==0:
    print(c)
  img_data = requests.get(image_url).content

  with open('/content/imgs/{}'.format(re.search(r'\w*\.\w*$', image_url).group(0)).lower(), 'wb') as handler:
      handler.write(img_data)
  img = '/content/imgs/{}'.format(re.search(r'\w*\.\w*$', image_url).group(0)).lower()
  result = inference_detector(model, img)

  if isinstance(result, tuple):
        bbox_result, segm_result = result
  else:
    bbox_result, segm_result = result, None
  bboxes = np.vstack(bbox_result)
  labels = [
    np.full(bbox.shape[0], i, dtype=np.int32)
    for i, bbox in enumerate(bbox_result)
    ]
  labels = np.concatenate(labels)
  assert len(labels)==len(bboxes)
  ind=0
  for i in range(len(labels)): 
    if bboxes[i,-1]<0.8 or labels[i]>26:
      continue

    else:
      if model.CLASSES[labels[i]] in clss[int(clf.predict([nm]))]: 
        cropped = mmcv.imcrop(mmcv.imread(img),bboxes[i][:-1])
        mmcv.imwrite(cropped,'/content/cropped/'+ img[14:-4]+'/'+str(ind)+'_'+model.CLASSES[labels[i]]+'_' + nm + '.jpg')

      else:
        continue
