import pandas as pd
from transformers import BertTokenizer
import tqdm
import torch 
from torch.utils.data import TensorDataset, random_split


def preprocess_data(data_dir1,data_dir2) :
    train_pro = pd.read_csv(data_dir1)
    train_sum = pd.read_csv(data_dir2)
    train_data = train_pro.merge(train_sum , on = "prompt_id")
    train_data.drop(["prompt_id" , "student_id"] , axis = 1 , inplace = True)

    # print(train_data.shape[0])    

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True,mirror='tuna')
    sentences = []
    labels_content = []
    labels_wording = []
    max_len = 0

    for index in tqdm.tqdm(range(train_data.shape[0]) , total = train_data.shape[0]):

        sent = ""
        for column in ["prompt_question" , "prompt_title" , "prompt_text" , "text"]:
            sent += str(train_data[column][index])
        sentences.append(sent)

        for column in ["content","wording"]:
            if column == "content" :
                labels_content.append(train_data[column][index])
            if column == "wording" :
                labels_wording.append(train_data[column][index])
                
        
        """
        #算max_len
        input_ids = tokenizer.encode(sent, add_special_tokens=True)
        max_len = max(max_len, len(input_ids))
        """
    # print(labels_content[0])
    # print(sentences[0])
    # print('Max sentence length: ', max_len)

    # 将数据集分完词后存储到列表中
    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # 输入文本
                            add_special_tokens = True, # 添加 '[CLS]' 和 '[SEP]'
                            max_length = 512,           # 填充 & 截断长度
                            padding='max_length',      # 填充到最大长度
                            truncation=True,          # 显式启用截断策略
                            return_attention_mask = True,   # 返回 attn. masks.
                            return_tensors = 'pt',     # 返回 pytorch tensors 格式的数据
                    )
        
        # 将编码后的文本加入到列表  
        input_ids.append(encoded_dict['input_ids'])
        
        # 将文本的 attention mask 也加入到 attention_masks 列表
        attention_masks.append(encoded_dict['attention_mask'])
    
    # 将列表转换为 tensor
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels_content = torch.tensor(labels_content)
    labels_wording = torch.tensor(labels_wording)

    # print(labels_content[0])
    # print('Original: ', sentences[0])                            
    # print('Token IDs:', input_ids[0])

    dataset = TensorDataset(input_ids, attention_masks, labels_content, labels_wording)
    # 计算训练集和验证集大小
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset,val_dataset