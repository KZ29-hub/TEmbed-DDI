from transformers import BioGptTokenizer, BioGptModel
import torch
import torch.nn as nn
import pandas as pd
# from googletrans import Translator
import pickle
import numpy as np
import torch.nn.functional as F
import random
import os

def eg(inputfile, outputfile, embedding_list):
    # load embeddings of all TCMs
    model_name = os.path.splitext(embedding_list)[0]
    embedding_file = os.path.join(inputfile, "ge1/llm_embedding/", embedding_list)
    print(f"embedding_file: {embedding_file}")

    with open(embedding_file, 'rb') as pkl_file:
        loaded_embeddings_dict_CH = pickle.load(pkl_file)
    # print(loaded_embeddings_dict_CH)
    print("all TCM embeddings:", len(loaded_embeddings_dict_CH))

############################################################################################################################
###### load pairs information
    DDI_file = os.path.join(inputfile, "data/", "TCM_DDI_Chinese.xlsx")
    print(f"DDI_file: {DDI_file}")
    df_pos = pd.read_excel(DDI_file)

    # load positive herb data
    herb_positive_dict = {}
    for index, row in df_pos.iterrows():
        key = row['中文名1']
        value = row['中文名2']
        if key not in herb_positive_dict:
            herb_positive_dict[key] = []
        herb_positive_dict[key].append(value)
        # herb_positive_dict[row['ephedra']] = row['Laurel branches']

    print("total positive dictionary items:", len(herb_positive_dict))
    total_positive_values = sum(len(value) for value in herb_positive_dict.values())
    print("total positive herb_couple items:", total_positive_values)
    # print("positive dictionary:", herb_positive_dict)

    # load pair
    DDI_neg_file = os.path.join(inputfile, "data/", "TCM_DDI_negative_Chinese.xlsx")
    df_neg = pd.read_excel(DDI_neg_file)
    # # load negative herb data
    herb_negative_dict = {}
    for index, row in df_neg.iterrows():
        # key = row['Chinese_name_neg1']
        # value = row['Chinese_name_neg2']
        key = row['中药中文名1']
        value = row['中药中文名2']
        if key not in herb_negative_dict:
            herb_negative_dict[key] = []
        herb_negative_dict[key].append(value)
        # herb_negative_dict[row['Chuanwu']] = row['Half summer']
    print("total negative dictionary items:",len(herb_negative_dict))
    total_negative_values = sum(len(value) for value in herb_negative_dict.values())
    print("total negative herb_couple items:", total_negative_values)
    # print("negative dictionary:", herb_negative_dict)


############################################################################################################################
    ####### embeddings concatenation  : Positive
    Combined_pos_dict_pin = {}
    print("!!!!!!!!!!!!!!!!!")
    print(herb_positive_dict)
    # print("!!!!!!!!!!!")
    # print(loaded_embeddings_negative_Chinese)
    # go through all the dict
    for key, values in herb_positive_dict.items():
        for value in values:
            matching_keys1 = [key_all1 for key_all1 in loaded_embeddings_dict_CH if key in key_all1]
            matching_keys2 = [key_all1 for key_all1 in loaded_embeddings_dict_CH if value in key_all1]
             # Get a vector of keys and a vector of values corresponding to the drug from dictionary
            # print(matching_keys1)
            # print(matching_keys2)
            if (len(matching_keys1) != 0) and (len(matching_keys2) != 0):
                for m1 in matching_keys1:
                    for m2 in matching_keys2:
                        herb1_pos_vector = loaded_embeddings_dict_CH[m1]
                        herb2_pos_vector = loaded_embeddings_dict_CH[m2]
                        # concatenation
                        print(herb1_pos_vector.shape)
                        print(herb2_pos_vector.shape)
                        concatenated_vector = torch.cat((torch.tensor(herb1_pos_vector).unsqueeze(0), torch.tensor(herb2_pos_vector).unsqueeze(0)), dim=1)
                        # shape
                        # print(concatenated_vector.shape)
                        # store to dict
                        concatenated_key = m1 + "," + m2
                        Combined_pos_dict_pin[concatenated_key] = concatenated_vector
    # print(Combined_pos_dict_pin)
    print(len(Combined_pos_dict_pin))
    
    # save
    oupath = os.path.join(outputfile, "eg2/", "concat_embeddings/", model_name)
    os.makedirs(oupath, exist_ok=True)
    
    with open(f"{oupath}/positive_chinese.pkl", 'wb') as pkl_file:
        pickle.dump(Combined_pos_dict_pin, pkl_file)
    
    print("embedding saved successfully")

# ############################################################################################################################
    ####### embeddings: negative
    Combined_pos_dict_pin = {}
    print("!!!!!!!!!!!!!!!!!")
    print(herb_negative_dict)
    # print("!!!!!!!!!!!")
    # print(loaded_embeddings_negative_Chinese)
    for key, values in herb_negative_dict.items():
        for value in values:
            matching_keys1 = [key_all1 for key_all1 in loaded_embeddings_dict_CH if key in key_all1]
            matching_keys2 = [key_all1 for key_all1 in loaded_embeddings_dict_CH if value in key_all1]
            print(matching_keys1)
            print(matching_keys2)
            if (len(matching_keys1) != 0) and (len(matching_keys2) != 0):
                for m1 in matching_keys1:
                    for m2 in matching_keys2:
                        herb1_pos_vector = loaded_embeddings_dict_CH[m1]
                        herb2_pos_vector = loaded_embeddings_dict_CH[m2]
                        # concat
                        concatenated_vector = torch.cat((torch.tensor(herb1_pos_vector).unsqueeze(0), torch.tensor(herb2_pos_vector).unsqueeze(0)), dim=1)
                        #
                        # print(concatenated_vector.shape)
                        # store to dict
                        concatenated_key = m1 + "," + m2
                        Combined_pos_dict_pin[concatenated_key] = concatenated_vector
    # print(Combined_pos_dict_pin)
    print(len(Combined_pos_dict_pin))

    # save
    with open(f"{oupath}/negative_chinese.pkl", 'wb') as pkl_file:
        pickle.dump(Combined_pos_dict_pin, pkl_file)

    print("embedding saved successfully")
    


############################################################################################################################
    ####### embeddings concatenation: undata
    undata_pairs_dict = {}
    Combined_un_dict_pin = {}
    # scores = []
    Combined_weight_dict = {}
    saved_keys = ""  # initialization
    
    # random select
    for _ in range(2000):
                # random select
                key1, key2 = random.sample(loaded_embeddings_dict_CH.keys(), 2)
                # save to saved_keys
                saved_keys += f"{key1},"
                saved_keys += f"{key2},"
                # print("saved_keys：", saved_keys)
    
                herb1_pos_vector = loaded_embeddings_dict_CH[key1]
                # print("herb1:", herb1_pos_vector.size)
                herb2_pos_vector = loaded_embeddings_dict_CH[key2]
                # concatenation
                concatenated_vector = torch.cat((torch.tensor(herb1_pos_vector).unsqueeze(0), torch.tensor(herb2_pos_vector).unsqueeze(0)), dim=1)
    
                # store to dict
                concatenated_key = key1 + "," + key2
                Combined_un_dict_pin[concatenated_key] = concatenated_vector
    
                # attention_scores = torch.dot(herb1_pos_vector, herb2_pos_vector)
                # # print(attention_scores)
                # scores.append(attention_scores)
    
    # print(Combined_un_dict_pin)
    print(len(Combined_un_dict_pin))
    
    # # save
    with open(f"{oupath}/undata2000_chinese.pkl", 'wb') as pkl_file:
        pickle.dump(Combined_un_dict_pin, pkl_file)
    
    print("embedding saved successfully")


if __name__ == '__main__':
    # "bge_embedding.pkl",
    inputfile = "/root/autodl-tmp/pycharmproject-herb1/"
    otuputfile = "/root/autodl-tmp/pycharmproject-herb1/"
    embedding_list = ["bge_embedding.pkl",  "me5_base_embedding.pkl", "me5_large_embedding.pkl", "me5_large_embedding.pkl", "qwen2_1.5B_embedding.pkl", "sfr1_embedding.pkl", "sfr2_embedding.pkl"]
    for embedding in embedding_list:
        print(f"---------------------------------------- {embedding} --------------------------------")
        eg(inputfile, otuputfile, embedding)
    print("All successful!!")