import torch
import torch.nn as nn
import pandas as pd
from transformers import RobertaTokenizer, TFRobertaModel
from tqdm import tqdm
import pickle

if __name__ == '__main__':

    def pooling(last_hidden_state: torch.Tensor,
                attention_mask: torch.Tensor=None):
        s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
        d = attention_mask.sum(dim=1, keepdim=True).float()
        return s / d

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_excel("/root/autodl-tmp/pycharmproject-herb1/data/Chinese_Pharmacopoeia_pieces_2015.xlsx")
    # concat columns
    selected_columns = ["Chinese_medicine_name", "Meridian_tropism_of_nature_and_flavor", "Functional_indications", "concocted", "Storage_method"]
    selected_columns_title = ["中药名"]
    data_dict = {}

    for index, row in df.iterrows():
        key = row[selected_columns_title[0]]
        value = " ".join([str(row[col]) for col in selected_columns[1:]])
        data_dict[key] = value
    print("total TCMs：", len(data_dict))

    # get keys and length
    keys = list(data_dict.keys())
    total_length = len(keys)

    tokenizer = RobertaTokenizer.from_pretrained('/root/autodl-tmp/models/roberta-base')
    model = TFRobertaModel.from_pretrained('/root/autodl-tmp/models/roberta-base')
    # model.to(device)

    print("Tokenizer and Model are downloaded success!!!")

    # get existing embeddings
    output_embeddings_dict = {}
    batch_size = 32  # Adjust the batch size as needed

    with torch.no_grad():
        for i in tqdm(range(0, len(data_dict), batch_size), total=len(data_dict) // batch_size):
            batch_keys = keys[i:i+batch_size]
            batch_values = [data_dict[key] for key in batch_keys]
            
            for idx, text in enumerate(batch_values):
                ### mean pooling
                inputs = tokenizer(text, return_tensors="tf")
                outputs = model(inputs)
                last_hidden_states = outputs.last_hidden_state
                embeddings = pooling(last_hidden_states, inputs['attention_mask'])

                output_embeddings_dict[batch_keys[idx]] = embeddings

    with open('/root/autodl-tmp/pycharmproject-herb1/ge1/llm_embedding/robert_base_embedding.pkl', 'wb') as pkl_file:
        pickle.dump(output_embeddings_dict, pkl_file)

    print("embedding saved successfully")