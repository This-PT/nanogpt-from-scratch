import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 32
batch_size = 16
learning_rate = 0.001
max_iters = 5000
eval_interval = 100
n_head = 4
dropout = 0
n_embd = 64
n_layer = 4
eval_iters = 100

with open('input.txt','r',encoding='utf-8') as f:
    test = f.read()

char_s = sorted(set(test))
vocab_size = len(char_s)

stoi = {ch:i for i,ch in enumerate(char_s)}
itos = {i:ch for i,ch in enumerate(char_s)}

encode = lambda s:[stoi[i] for i in s]
decode = lambda s:"".join([itos[i] for i in s])

df = torch.tensor(encode(test))

df_train = df[:int(len(df)*0.9)]
df_val = df[int(len(df)*0.9):]


class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd,head_size,bias=False)
        self.query = nn.Linear(n_embd,head_size,bias=False)
        self.value = nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B,T,C = x.shape
        k =  self.key(x)
        q = self.query(x)

        wei = q@k.transpose(-2,-1)*C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0,float('-inf'))
        wei = F.softmax(wei,dim=-1)
        sei = self.dropout(wei)

        v = self.value(x)
        out = wei@v
        return out

class MultiheadAttention(nn.Module):

    def __init__(self,num_head,head_size):
        super().__init__()
        self.head = nn.ModuleList([Head(head_size) for _ in range(num_head)])
        self.proj = nn.Linear(n_embd,n_embd)
        self.dropout = nn.Dropout(dropout)


    def forward(self,x):
        out = torch.cat([h(x)for h in self.head],dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,n_embd*4),
            nn.ReLU(),
            nn.Linear(n_embd*4,n_embd),
            nn.Dropout(dropout),
        )

    def forward(self,x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self,n_embd,n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiheadAttention(n_head,head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self,x):
        x = x+self.sa(self.ln1(x))
        x = x+self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        self.block = nn.Sequential(*[Block(n_embd,n_head)for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)

    def forward(self,idx,targets = None):
        B,T = idx.size()

        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T,device=device))
        x = token_emb+pos_emb
        x = self.block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)


        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits,loss

    def generate(self,idx,max_new_tokens):

        for _ in range(max_new_tokens):
            idx_cond = idx[:,-block_size:]
            logits,loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=-1)
        return idx

def get_batch(split):
    d = df_train if split == 'train ' else df_val
    xi =  torch.randint(len(d)-block_size , (batch_size,))
    x = torch.stack([d[i:i+block_size]for i in xi])
    y = torch.stack([d[i+1:i+block_size+1]for i in xi])
    x, y = x.to(device), y.to(device)
    return x, y



def estimate_loss():
    out={}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits,loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out





model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

for iter in range(max_iters):
    if(iter % eval_interval == 0 or iter == max_iters-1):
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb,yb = get_batch('train')

    logits,loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context,2000)[0].tolist()))




