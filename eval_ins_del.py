import numpy as np
import glob
import pdb
import os
from PIL import Image
from evaluation import CausalMetric, auc, gkern
import torch
from model import DenseNet121
#from covid_model import DenseNet121
import torch.nn as nn
from collections import OrderedDict
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from tqdm import tqdm 
import json
import math
dataset_type = 'covid'
model_weights = "/data/brian.hu/covid_saliency/covid_densenet121_embed_256_seed_1_epoch_20_ckpt.pth"


class InsDel():
    def __init__(self,
                model):
        self.model = model
        net_in_size = 224
        klen = 51
        ksig = math.sqrt(50)
        kern = gkern(klen, ksig)
        blur = lambda x: nn.functional.conv2d(x, kern.cuda(), padding=klen//2)
        self.insertion = CausalMetric(self.model, 'ins', net_in_size, substrate_fn=blur)
        self.deletion = CausalMetric(self.model, 'del', net_in_size, substrate_fn=torch.zeros_like)
        
    def evaluate(self, new_sal, ret_image):
        """
        This function evaluates an image and its saliency map for deletion 
        and insertion.
        
        Attributes:
            new_sal (numpy.array): The saliency map obtained at iteration k. 
            ret_image (torch.tensor.cuda): Input image that needs to be explained.
            
        Returns:
        score_del (float): The deletion score between 0-1 range.
        score_ins (float): The insertion score between 0-1 range.
        """
        new_sal = torch.from_numpy(new_sal).float()
        score_del = 0
        zero_cnt_del = 0
        score_del, zero_cnt_ins = self.deletion.single_run(self.q_image.cuda(), \
                                                           ret_image.cuda(), new_sal, verbose=0)
        score_ins, zero_cnt_del = self.insertion.single_run(self.q_image.cuda(), \
                                                            ret_image.cuda(),  new_sal, verbose=0)
        return score_del, score_ins, zero_cnt_ins, zero_cnt_del
    
    def load_query(self, query_image):
        self.q_image = query_image
    
    def forward(self, q_image, ret_dict, sal_dict):
        ins_avg = []
        del_avg = []
        z_ins_list = []
        z_del_list = []

        self.load_query(q_image)
        for i, sal_m in enumerate(sal_dict):
            ret_image = ret_dict[i]
            dele, ins, z_ins, z_del = self.evaluate(sal_m, ret_image )
            z_ins_list.append(z_ins)
            z_del_list.append(z_del)
            ins_avg.append(ins)
            del_avg.append(dele)
        return ins_avg, del_avg, z_ins_list, z_del_list 

        
# Class for average counter
class AverageCounter():
    def __init__(self):
        self.average = 0
        self.running_avg = 0
        self.fina_dict = {}
    
    def store(self, q_label, metric,  k=20):
        def check_if_in_dict(value1):
            if value1 in self.fina_dict.keys():
                return True
            else:
                return False
        if check_if_in_dict(q_label):
            self.fina_dict[q_label].append(metric)
        else:
            self.fina_dict[q_label] = [metric]
    
    def read_Average(self):
        avgdict = {}
        for k,v in self.fina_dict.items():
            avgdict[k] = sum(v)/ float(len(v))
        return avgdict
            
def prep_image_(file_n):
    query_image = Image.open(os.path.join(query_img_path, file_n)).convert('RGB')
    query_image_tensor = transform(query_image).unsqueeze_(0).cuda()
    return query_image_tensor
            
model = DenseNet121()
model.load_state_dict(torch.load(model_weights), strict=False)
#model = nn.Sequential(*list(model.children())[:-1])
model = model.eval()
model = model.cuda()

# Logging counter
ins_avg_c = AverageCounter()
del_avg_c = AverageCounter()



sal_map = {}
ret_map = {}
query_image_list = []
class_labels = {}

if dataset_type == 'covid':
    main_path = '/data/brian.hu/covid_saliency/simatt/'
    query_img_path = '/data/brian.hu/COVID/data/test/'
    valid_class = ['pneumonia', 'normal']
    # Unravel labels here
    with open('test_COVIDx4.txt', 'r') as f:
        for line in f.readlines():
            label = line.split()[2]
            if label not in valid_class:
                label = 'covid'
            #q_na = line.split()[1].split('.')[0]
            #class_labels[q_na.split('.')[0]] = label
            q_na = line.split()[1]
            class_labels[q_na] = label
            
if dataset_type == 'isic':
    main_path = '/data/xai/isic_saliency/sbsm'
    query_img_path = '/data/xai/ISIC-2017_Test_v2_Data'
    import pandas as pd
    valid_class = ['melanoma', 'seborrheic_keratosis']
    df = pd.read_csv('./ISIC-2017_Test_v2_Part3_GroundTruth_balanced.csv')
    for kj, im_name in enumerate(df['image_id']):
        if (df['melanoma'][kj] + df['seborrheic_keratosis'][kj]) > 0:
            if df['melanoma'][kj]:
                class_labels[im_name+'.jpg'] = 'melanoma'
            else:
                class_labels[im_name+'.jpg'] = 'seborrheic_keratosis'
        else:
            class_labels[im_name+'.jpg'] = 'nevi'

transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

f = open('./inser_dele_wacv_test_simatt.json','w')
f2 = open('./key_list_wacv_test_simatt.json','w')
ins_del_q_dict = {}
key_dict = {}
get_insert_dele = InsDel(model)
for file_n in os.listdir(main_path):
    print(file_n)
    query_image_tensor = prep_image_(file_n)
    retrieval_names = os.listdir(os.path.join(main_path, file_n))
    for r_n in retrieval_names:
        try:
            key_dict[file_n.split('/')[0].split('.')[0]] = [r_n]
            sal_map[file_n.split('/')[0].split('.')[0]].append(np.load(os.path.join(main_path, file_n,r_n)))
            ret_map[file_n.split('/')[0].split('.')[0]].append(prep_image_(os.path.join(query_img_path, \
                                                                '.'.join(r_n.split('.')[:-1]))))
        except KeyError:
            key_dict[file_n.split('/')[0].split('.')[0]].append(r_n)
            sal_map[file_n.split('/')[0].split('.')[0]] = [np.load(os.path.join(main_path, file_n,r_n))]
            ret_map[file_n.split('/')[0].split('.')[0]] = [prep_image_(os.path.join(query_img_path, \
                                                                '.'.join(r_n.split('.')[:-1])))]

    # Compute local deletion and insertion metric 
    insertion, deletion, i_in, i_del = get_insert_dele.forward(query_image_tensor, ret_map[file_n.split('/')[0].split('.')[0]],sal_map[file_n.split('/')[0].split('.')[0]])

    avg_insert, avg_del =  (sum(insertion)/len(insertion)), (sum(deletion)/len(deletion))
    # Aggerate calculate metric to associated class
    print(avg_insert, avg_del)
    try:
        assert ins_del_q_dict[file_n.split('/')[0].split('.')[0]]
        ins_del_q_dict[file_n.split('/')[0].split('.')[0]].append([insertion, deletion])
    except KeyError:
         ins_del_q_dict[file_n.split('/')[0].split('.')[0]] = [insertion, deletion] 
    ins_avg_c.store(class_labels[file_n], avg_insert)
    del_avg_c.store(class_labels[file_n], avg_del)
    del sal_map[file_n.split('/')[0].split('.')[0]] 
json.dump(ins_del_q_dict, f)
json.dump(key_dict, f2)
print(ins_avg_c.read_Average())
print(del_avg_c.read_Average())
