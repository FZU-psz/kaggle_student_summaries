import torch 
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
from preprocess import preprocess_data
from model import BertForRegression
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # 四舍五入到最近的秒
    elapsed_rounded = int(round((elapsed)))
    
    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

data_dir1="./data/prompts_train.csv"
data_dir2="./data/summaries_train.csv"
batch_size = 16
train_dataset,val_dataset=preprocess_data(data_dir1,data_dir2)

# 为训练和验证集创建 Dataloader，对训练样本随机洗牌
train_dataloader = DataLoader(
            train_dataset,  # 训练样本
            sampler = RandomSampler(train_dataset), # 随机小批量
            batch_size = batch_size, # 以小批量进行训练
        )

# 验证集不需要随机化，这里顺序读取就好
validation_dataloader = DataLoader(
            val_dataset, # 验证样本
            sampler = SequentialSampler(val_dataset), # 顺序选取小批量
            batch_size = batch_size 
        )

bert_model_name='bert-base-uncased'
model=BertForRegression(bert_model_name)
# 在 gpu 中运行该模型
model.cuda()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
num_epochs = 5

# 数据处理和加载
train_texts = []
train_content_scores = []
train_wording_scores = []

def train_model(model, train_loader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, num_epochs))
        print('Training...')
        # 统计单次 epoch 的训练时间
        t0 = time.time()
        # 重置每次 epoch 的训练总 loss
        total_loss = 0.0
        # 训练集小批量迭代
        for step, batch in enumerate(train_loader):
            input_ids, attention_mask, labels_content, labels_word = [item.to(device) for item in batch]
            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits.squeeze(1), labels_content) + criterion(logits.squeeze(1), labels_word)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 每经过40次迭代，就输出进度信息
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_loader), elapsed))

        # 单次 epoch 的训练时长
        training_time = format_time(time.time() - t0)        
        avg_loss = total_loss / len(train_loader)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_loss))
        print("  Training epcoh took: {:}".format(training_time))

# 开始训练
train_model(model, train_dataloader, optimizer, criterion, num_epochs)