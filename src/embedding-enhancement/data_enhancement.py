import pickle #用于字典数据保存
import random
import torch
import torch.nn as nn
# from transformers import BertModel, BertTokenizer
import torch.optim as optim
from torch.nn import Transformer
import numpy as np
import os

def neg_aug(ou_file):
    ######negative
    for dirpath, dirnames, filenames in os.walk(ou_file):
        for dirname in dirnames:
            neg_path = os.path.join(ou_file, dirname, "negative_chinese.pkl")
            ou_neg_path = os.path.join(ou_file, dirname, "negative_chinese_aug.pkl")
    
            with open(neg_path, 'rb') as pkl_file:
                loaded_negative_pin_dict = pickle.load(pkl_file)
            print("embedding showed successfully: loaded_negative_pin_dict")
            # print(loaded_negative_pin_dict)
            print("loaded_negative_pin_dict length:", len(loaded_negative_pin_dict))
        
        
            def random_perturbation(embeddings, stddev=0.15):
                perturbed_embeddings = embeddings + torch.randn_like(embeddings) * stddev
                return perturbed_embeddings
        
            # negative disturbation
            loaded_pin_dict_addpert_non = {}
            for key, value in loaded_negative_pin_dict.items():
                perturbed_embedding = random_perturbation(value)
                loaded_pin_dict_addpert_non[key] = loaded_negative_pin_dict[key]
                # modify keys
                new_key = key + '_pertured'
                # modify embeddings
                loaded_pin_dict_addpert_non[new_key] = perturbed_embedding
            print(len(loaded_pin_dict_addpert_non))
        
            # save
            with open(ou_neg_path, 'wb') as pkl_file:
                print("ou_neg:", ou_neg_path)
                pickle.dump(loaded_pin_dict_addpert_non, pkl_file)

    print("All embedding augmentation successful!!!")

if __name__ == '__main__':
    ou_file = "/root/autodl-tmp/pycharmproject-herb1/eg2/concat_embeddings/"
    neg_aug(ou_file)