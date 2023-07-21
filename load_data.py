# 大数据 彭诗忠
#开发时间  20:00 2023/7/21

import torch

data_path = '..\data\data_dict.pt'
def load_data():
    data_dict= torch.load(data_path)
    input_ids= data_dict['input_ids']
    attention_masks = data_dict['attention_masks']
    labels_content=data_dict['label_content']
    labels_wording = data_dict['label_wording']

    return input_ids,attention_masks,labels_content,labels_wording

# input_ids,_,_,_ = load_data()
# print(input_ids)