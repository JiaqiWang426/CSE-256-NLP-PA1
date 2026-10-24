'''
你对 SentimentExample 的理解正确，对 WordEmbeddings 的理解基本正确。主要需要修正最后一点：这里的 Indexer 是在加载 GloVe 时建立的，
不会针对每个句子重新建立。
这两个文件一共定义了三个类：
类	一个对象代表什么	保存的主要内容
SentimentExample	一条带标签的句子样本	words：字符串列表；label：整数 0 或 1
Indexer	一份对象与整数 ID 的双向映射表	objs_to_ints 和 ints_to_objs
WordEmbeddings	加载到内存中的一套词向量及其查询表	word_indexer：一个 Indexer 实例；vectors：向量矩阵


它们的关系可以分成两条数据读取流程：
train.txt / dev.txt
        │ read_sentiment_examples()
        ▼
多个 SentimentExample 对象
每个保存 words 和 label


GloVe 文件
        │ read_word_embeddings()
        ▼
一个 WordEmbeddings 对象
        ├── word_indexer：一个 Indexer 对象
        │       ├── objs_to_ints：单词 → ID
        │       └── ints_to_objs：ID → 单词
        └── vectors：第 i 行是 ID 为 i 的单词的向量
这两个流程得到的对象，是在处理句子时通过“单词”联系起来的。 例如：
# 一条样本
example.words = ["good", "movie"]
example.label = 1

# 共用的词向量对象，假设加载后有以下映射
embeddings.word_indexer.index_of("good")   # 例如返回 25
embeddings.word_indexer.index_of("movie")  # 例如返回 80

# 对应词向量
embeddings.vectors[25]  # good 的向量
embeddings.vectors[80]  # movie 的向量
因此，你最后一句应改成：
加载 GloVe 文件时，代码创建一个共用的 Indexer，为其中每个单词分配 ID。处理每个 SentimentExample 时，用它的 words 查询已有的 ID，再根据 ID 获取向量。

另外两个小细节：
- 类名是 WordEmbeddings，不是 WorldEmbeddings。它代表文件加载后的数据，不是文件本身。
- Indexer(object) 中的 object 表示继承 Python 的基础类。Indexer 本身是通用映射工具；在这里，它被用来映射单词与 ID。ID 对应加载后 vectors 的行号，
包含额外加入的 PAD 和 UNK，所以不直接等于原始 GloVe 文件的行号。
'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import math
from utils import *
from sentiment_data import *

########################
#Training and Evaluation
#The training and back propagation
def training_and_evaluation(X_train, y_train, X_dev, y_dev, model,
                            optimizer, criterion, epochs=100, eval_every=5,
                            batch_size=None):
    """Train the model and record metrics every eval_every epochs and at the end."""
    if epochs < 1 or eval_every < 1:
        raise ValueError("epochs and eval_every must be positive")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive")
    # Preserve the original full-batch path unless a batch size is supplied.
    if batch_size is None:
        batches = ((X_train, y_train),)
    else:
        batches = DataLoader(TensorDataset(X_train, y_train),
                             batch_size=batch_size, shuffle=True)

    history = {
        "epoch": [],
        "train_loss": [],
        "dev_loss": [],
        "train_acc": [],
        "dev_acc": []
    }
    for epoch in range(1, epochs + 1):
        # evaluation() switches to eval mode, so restore dropout each epoch.
        model.train()
        for X_batch, y_batch in batches:
            optimizer.zero_grad()
            train_logits = model(X_batch)
            loss = criterion(train_logits, y_batch)
            loss.backward()
            optimizer.step()

        if epoch % eval_every == 0 or epoch == epochs:
            # Measure both datasets after the update, with dropout disabled.
            train_loss, train_acc = evaluation(X_train, y_train, model, criterion, batch_size)
            dev_loss, dev_acc = evaluation(X_dev, y_dev, model, criterion, batch_size)
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["dev_loss"].append(dev_loss)
            history["train_acc"].append(train_acc)
            history["dev_acc"].append(dev_acc)
            print(
                f"Epoch {epoch}: train loss {train_loss:.4f}, "
                f"dev loss {dev_loss:.4f}, train accuracy {train_acc:.4f}, "
                f"dev accuracy {dev_acc:.4f}"
            )
    return history
#Evaluation
def evaluation(X, y, model, criterion, batch_size=None):
    """Evaluate mean-reduced loss and accuracy, optionally in batches."""
    model.eval()
    if batch_size is None:
        batch_size = len(y)
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(y), batch_size):
            X_batch = X[start:start + batch_size]
            y_batch = y[start:start + batch_size]
            logits = model(X_batch)
            total_loss += criterion(logits, y_batch).item() * len(y_batch)
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
    return total_loss / len(y), correct / len(y)

########################
#Part 1A: Use glove
#Then I do mean-pooling to each sentence in train.txt
def glove_read_X_y(Embeddings, dim, infile: str):
    exs = read_sentiment_examples(infile)
    word_indexer = Embeddings.word_indexer
    unk_index = word_indexer.index_of("UNK") #which is 1
    X, y = [], []
    for sentiment in exs:
        words = sentiment.words
        x = torch.zeros(dim)
        for word in words:
            index = word_indexer.index_of(word)
            if index == -1:
                index = unk_index 
            vector = torch.as_tensor(
                    Embeddings.vectors[index], dtype=torch.float32
                )
            x += vector
        x = x/max(len(words), 1)
        X.append(x)
        y.append(sentiment.label)
    return torch.stack(X), torch.tensor(y, dtype=torch.long)

#Next I define the neural network
class glove_TwoLayerMLP(nn.Module):
    def __init__(self, dim):
        super().__init__() #Suppose the hidden layer is 100
        self.layer1 = nn.Linear(in_features=dim, out_features=100, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.layer2 = nn.Linear(in_features=100, out_features=2, bias=True)

    def forward(self, x): #a binary predictor; output_features = 2
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

########################
#Part 1B: Train embeddings
#Then I do mean-pooling to each sentence in train.txt
def embeddings_read_X_y(Embeddings, infile: str):
    exs = read_sentiment_examples(infile)
    word_indexer = Embeddings.word_indexer
    unk_index = word_indexer.index_of("UNK") #which is 1
    X, y = [], []
    max_length = max(len(sentiment.words) for sentiment in exs)
    for sentiment in exs:
        words = sentiment.words
        x = []
        for word in words:
            index = word_indexer.index_of(word)
            if index == -1:
                index = unk_index 
            x.append(index)
        x += [0] * (max_length - len(x)) #use PAD index to fill the blanks
        X.append(x)
        y.append(sentiment.label)
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

#Next I define the neural network
class embeddings_TwoLayerMLP(nn.Module):
    def __init__(self, vocab_size):
        super().__init__() #Embed each
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=300, #same as glove
            padding_idx = 0
        ) #size: vocab * 300
        self.layer1 = nn.Linear(in_features=300, out_features=100, bias=True) #vocab * 100
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.layer2 = nn.Linear(in_features=100, out_features=2, bias=True) #vocab * 2

    def embed(self, x):
        mask = (x != 0).unsqueeze(-1) # [B, L, 1]
        embedded = self.embedding(x) # [B, L, 300]
        summed = (embedded * mask).sum(dim=1) #[B, 300]
        lengths = mask.sum(dim=1).clamp(min=1) #[B, 1]
        return summed / lengths #[B, 300]


    def forward(self, x): #a binary predictor; output_features = 2
        #x size: batch * 300
        x = self.embed(x)
        x = self.layer1(x) #batch * 100
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x) #batch * 2
        return x
