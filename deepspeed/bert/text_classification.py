import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer


def read_data(file_name, num=180000):
    all_label = []
    all_text = []

    with open(file_name, encoding="uft-8") as f:
        # avoid empty string
        all_data = f.read().split("\n")[:-1][:num]

    for text in all_data:
        text_, label_ = text.split("\t")

        all_text.append(text_)
        all_label.append(label_)

    return all_text, all_label


class MyDataset(Dataset):

    def __init__(self, all_text, all_label):
        self.all_text = all_text
        self.all_label = all_label

        self.tokenizer = BertTokenizer.from_pretrained("bert_base_chinese")

    def __getitem__(self, index):
        text = self.all_text[index]
        label = self.all_label[index]

        text_index = self.tokenizer.encode(text, truncation=True, padding="max_length", max_length=15)

        return torch.tensor(text_index), torch.tensor(label)

    def __len__(self):
        return len(self.all_text)


class BertClassifier(nn.Module):

    def __init__(self, class_num):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert_base_chinese")

        for name, param in self.bert.named_parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(786, class_num)
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, batch_text, batch_label=None):
        bert_out = self.bert.forward(batch_text, attention_mask=batch_text > 0)

        bert_out1, bert_out2 = bert_out[0], bert_out[1]
        pre = self.classifier(bert_out2)

        if batch_label is not None:
            loss = self.loss_fun(pre, batch_label)
            return loss
        else:
            return torch.argmax(pre, dim=-1)


if __name__ == "__main__":
    train_text, train_label = read_data("data/train.txt")
    dev_text, dev_label = read_data("data/dev.txt", )

    epoch = 20
    batch_size = 2
    class_num = len(set(train_label))
    lr = 0.0001
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_dataset = MyDataset(train_text, train_label)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    dev_dataset = MyDataset(dev_text, dev_label)
    dev_dataloader = DataLoader(dev_dataset, batch_size=10, shuffle=False)

    model = BertClassifier(class_num).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr)

    for e in range(epoch):
        model.train()

        for b_i, (batch_text, batch_label) in enumerate(train_dataloader):
            batch_text = batch_text.to(device)
            batch_label = batch_label.to(device)
            loss = model.forward(batch_text, batch_label)
            loss.backward()
            optim.step()
            optim.zero_grad()

            if b_i % 30 == 0:
                print(f"Loss:{loss:.2f}")

        right = 0
        model.eval()
        for batch_text, batch_label in dev_dataloader:
            batch_text = batch_text.to(device)
            batch_label = batch_label.to(device)
            pre = model.forward(batch_text)
            right += int(sum(batch_label == pre))
        print(f"acc={right / len(dev_text) * 100} %")
