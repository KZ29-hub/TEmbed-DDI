import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, average_precision_score
from transformers import BioGptTokenizer, BioGptForCausalLM
from torch import Tensor
from transformers import BioGptTokenizer, BioGptModel, AutoModel, AutoTokenizer
import torch.nn.functional as F
from data_utils import *
from tqdm import tqdm
import os
import argparse
import random

class MainModel(nn.Module):
    def last_token_pool(self, last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
        
    def __init__(self, embedding_dim):
        super(MainModel, self).__init__()

        # Load the BioGPT tokenizer and model
        self.tokenizer = BioGptTokenizer.from_pretrained("/root/autodl-tmp/multiview-project-1023/pretrained_models/biogpt")
        self.model = BioGptModel.from_pretrained("/root/autodl-tmp/multiview-project-1023/pretrained_models/biogpt")


        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool1d(kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)

        self.bilstm1 = nn.LSTM(2048, 1024, batch_first=True, bidirectional=True)
        # self.bilstm2 = nn.LSTM(2048, 512, batch_first=True, bidirectional=True)

        self.flatten = nn.Flatten()

        self.fc_final = nn.Linear(2048 * 64, 2)
        # # 设置L2正则化的权重
        # self.l2_regularization = 0.001

        # Self-Attention layer
        self.self_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=8)

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim , nhead=8)
        self.transformer1 = nn.TransformerEncoder(encoder_layers, num_layers=4)

        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim , nhead=8)
        self.transformer2 = nn.TransformerEncoder(encoder_layers, num_layers=4)

    def forward(self, data):
        # Extract data
        desc1 = [data[i]['drug1_name'] + data[i]['drug1_smile'] + data[i]['drug1_desc'] for i in range(len(data))]
        desc2 = [data[i]['drug2_name'] + data[i]['drug2_smile'] + data[i]['drug2_desc'] for i in range(len(data))]

        # Generate embeddings for descriptions
        encoded_desc1 = self.tokenizer(desc1, return_tensors='pt', padding=True, truncation=True).to(device)
        encoded_desc2 = self.tokenizer(desc2, return_tensors='pt', padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            output_desc1 = self.model(**encoded_desc1)
            output_desc2 = self.model(**encoded_desc2)
       
        # print('last_hidden_state', output_desc1.last_hidden_state.size())

        transformer1_embed = self.transformer1(output_desc1.last_hidden_state)
        transformer2_embed = self.transformer2(output_desc2.last_hidden_state)
        # print('transformer1_embed', transformer1_embed.size())
        # print('transformer2_embed', transformer2_embed.size())
        
        attention_output1, _ = self.self_attention(transformer2_embed.transpose(0, 1), transformer1_embed.transpose(0, 1), transformer1_embed.transpose(0, 1))
        # print('transformer1_embed', attention_output1.size())
        x1 = self.last_token_pool(attention_output1.transpose(0, 1), encoded_desc2['attention_mask'])
        x1 = F.normalize(x1, p=2, dim=1)
        
        attention_output2, _ = self.self_attention(transformer1_embed.transpose(0, 1), transformer2_embed.transpose(0, 1), transformer2_embed.transpose(0, 1))
        x2 = self.last_token_pool(attention_output2.transpose(0, 1), encoded_desc1['attention_mask'])
        x2 = F.normalize(x2, p=2, dim=1)
        x = torch.cat((x1, x2), dim=1)
        
        # # # Use the last hidden state of the last token (CLS token) as the embedding
        # embeddings = self.last_token_pool(transformer1_embed, encoded_desc1['attention_mask'])
        # desc_embedding1 = F.normalize(embeddings, p=2, dim=1)
        # embeddings2 = self.last_token_pool(transformer2_embed, encoded_desc2['attention_mask'])
        # desc_embedding2 = F.normalize(embeddings2, p=2, dim=1)

        # x = torch.cat((desc_embedding1, desc_embedding2), dim=1)

        
        ## 注意力机制
        # x, _ = self.self_attention(x, x, x)
        
        ## 拼接
        # x = torch.cat((desc_embedding1, desc_embedding2), dim=1)
        
        # print('x', x.size())
        x = x.unsqueeze(1)  # Add a channel dimension
        # print("initial", x.size())
        x = self.conv1(x)
        # print('conv1', x.size())
        x = self.relu(x)
        # print('relu', x.size())

        # 第二层
        x = self.conv2(x)
        # print('conv1', x.size())
        x = self.relu(x)
        # print('relu', x.size())

        # # First BiLSTM layer
        # x, _ = self.bilstm1(x)
        # # print('first lstm size', x.size())

        # # Second BiLSTM layer
        # x, _ = self.bilstm2(x)
        # # print('second lstm size', x.size())

        x = self.flatten(x)
        # print('flatten', x.size())
        x = x.squeeze(0)  # Remove sequence length dimension
        # print('flatten lstm size', x.size())
        x = self.fc_final(x)
        # print("x", x.shape)
        return x

# Training example
### weight_decay=1e-5
def train(model, data_list, labels, test_data_list, test_labels, csv_file, epochs=10, batch_size=32, lr=2e-5, dataset_name='unknown'):
    # Calculate class weights
    total_samples = len(labels)
    class_counts = [0, 0]
    
    for value in labels:
        class_counts[value] += 1

    class_weights = torch.tensor([
        total_samples / max(class_counts[0], 1),
        total_samples / max(class_counts[1], 1),
    ]).to(device)
    print('class_weight:', class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # Use Cross Entropy Loss for multi-class classification
   
    
    for epoch in range(epochs):
        # # Set the learning rate
        # if epoch < growth_epochs:
        #     lr = (max_lr / growth_epochs) * (epoch + 1)  # Linear growth
        # else:
        #     lr = max_lr  # Maintain max_lr

        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
        # print('lr', lr)
        
        model.train()
        total_loss = 0
        for i in tqdm(range(0, len(data_list), batch_size), total=len(data_list)//batch_size,desc=f'{epoch} training'):
            batch_data = data_list[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]

            # 将列表转换为张量，并确保是长整型
            batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(next(model.parameters()).device)
            batch_labels = batch_labels.squeeze()  # 使其形状变为 [32]

            # print("batch_labels", batch_labels.shape)
            
            optimizer.zero_grad()
            output = model(batch_data)  # Ensure the output shape matches the number of classes
            loss = criterion(output, batch_labels.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # print(f'Epoch {epoch + 1}, Batch Loss: {loss.item()}')
        
        average_loss = total_loss / (len(data_list) // batch_size)
        print(f'Epoch {epoch} Total Loss: {average_loss}')

        

        # Perform testing at specified epochs
        if epoch in [0, 2, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49] or epoch == epochs - 1:  # Test at 10, 20, 30 and last epoch
            test_acc, test_precision, test_recall, test_f1, test_auc, test_aupr = test(model, test_data_list, test_labels, batch_size)
            new_item = {
                'method': f'{dataset_name}_ep{epochs}_bs{batch_size}_lr{lr}_biogpt_cfa_fulldata_balanced',
                'epoch': epoch + 1,  # Storing 1-based epoch number
                'accuracy': test_acc,
                'precision': test_precision,
                'recall': test_recall,
                'f1_score': test_f1,
                'auc': test_auc,
                'aupr':test_aupr
            }
            if os.path.exists(csv_file):
                writeCSV_xu([new_item], csv_file)
            else:
                writeCSV([new_item], csv_file)



def test(model, test_data_list, test_labels, batch_size=32):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, len(test_data_list), batch_size),total=len(test_data_list)//batch_size,desc='testing'):
            batch_data = test_data_list[i:i + batch_size]

            output = model(batch_data)
            # print('output', output)
            # print('output', output.tolist())
            # predictions.extend(output.tolist())
    
            # Check if output is 1D or 2D and adjust accordingly
            if output.ndimension() == 1:  # If output is 1D
                predictions.append(output.tolist())
            else:  # If output is 2D
                predictions.extend(output.tolist())
            
    
    # Convert predictions to binary
    print('predictions', predictions)
    predictions2 = [1 if pred[1] >= pred[0] else 0 for pred in predictions]
    print('predictions2', predictions2)
    true_labels = test_labels.squeeze().tolist()

    # # Calculate metrics
    # acc = accuracy_score(true_labels, predictions2)
    # precision = precision_score(true_labels, predictions2, average='macro')
    # recall = recall_score(true_labels, predictions2, average='macro')
    # f1 = f1_score(true_labels, predictions2, average='macro')

    # Calculate metrics
    acc = accuracy_score(true_labels, predictions2)
    precision = precision_score(true_labels, predictions2, average='macro')
    recall = recall_score(true_labels, predictions2, average='macro')
    f1 = f1_score(true_labels, predictions2, average='macro')

    # Calculate AUC and AUPR
    if len(set(true_labels)) == 2:  # Check if binary classification
        auc = roc_auc_score(true_labels, [prob[1] for prob in predictions])  # Using the probabilities of the positive class
        aupr = average_precision_score(true_labels, [prob[1] for prob in predictions])
    else:
        auc = None
        aupr = None

    print(f'Accuracy: {acc:.4f}')
    print(f'Precision (macro): {precision:.4f}')
    print(f'Recall (macro): {recall:.4f}')
    print(f'F1 Score (macro): {f1:.4f}')
    if auc is not None:
        print(f'AUC: {auc:.4f}')
        print(f'AUPR: {aupr:.4f}')

    return acc, precision, recall, f1, auc, aupr

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a model with specified parameters.')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the training data')
    parser.add_argument('--test_path', type=str, required=True, help='Path to the testing data')
    parser.add_argument('--epoch', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=2e-5, help='Maximum learning rate')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the results')
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset_name')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_json = readJSONL(args.train_path)
    test_json = readJSONL(args.test_path)

    random.seed(42)
    random.shuffle(train_json)
    
    train_labels = []
    for example in train_json:
        if example["pos"] == 'advise':
            train_labels.append(1)
        else:
            train_labels.append(0)
    test_labels = []
    for example in test_json:
        if example["pos"] == 'advise':
            test_labels.append(1)
        else:
            test_labels.append(0)
    
    train_true_labels = torch.tensor(train_labels).to(device)
    test_true_labels = torch.tensor(test_labels).to(device)
    

    embedding_dim = 1024  # Set according to the BioGPT model output dimension
    model = MainModel(embedding_dim).to(device)
    train(model, train_json, train_true_labels, test_json, test_true_labels, args.csv_file, args.epoch, args.batch_size, args.lr, args.dataset_name)
    # test(model, test_json[:2000], test_true_labels[:2000], batch_size)

