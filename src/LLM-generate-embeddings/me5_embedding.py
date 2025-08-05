import torch
import torch.nn as nn
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle

if __name__ == '__main__':
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

    # biogpt model
    model = SentenceTransformer('/root/autodl-tmp/models/multilingual-e5-large')
    model.to(device)
    print("Tokenizer and Model are downloaded success!!!")

    # get existing embeddings
    output_embeddings_dict = {}
    batch_size = 32  # Adjust the batch size as needed

    with torch.no_grad():
        for i in tqdm(range(0, len(data_dict), batch_size), total=len(data_dict) // batch_size):
            batch_keys = keys[i:i+batch_size]
            batch_values = [data_dict[key] for key in batch_keys]
            
            for idx, text in enumerate(batch_values):
                embeddings = model.encode(text, normalize_embeddings=True)
                output_embeddings_dict[batch_keys[idx]] = embeddings

    with open('/root/autodl-tmp/pycharmproject-herb1/ge1/llm_embedding/me5_large_embedding.pkl', 'wb') as pkl_file:
        pickle.dump(output_embeddings_dict, pkl_file)

    print("embedding saved successfully")