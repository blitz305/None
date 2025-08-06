import torch
import torch.nn as nn
import torch.nn.functional as F

class MemN2N(nn.Module):
    def __init__(self, vocab_size, embedding_size, memory_size, sentence_size, hops=3):
        super(MemN2N, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.memory_size = memory_size
        self.sentence_size = sentence_size
        self.hops = hops

        # A embedding: memory encoder
        self.A = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=0)
        # C embedding: memory output
        self.C = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=0)
        # B embedding: query encoder
        self.B = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=0)

        self.softmax = nn.Softmax(dim=1)

        # 初始化为可训练参数
        for emb in [self.A, self.B, self.C]:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, story, query):
        """
        story: [batch_size, memory_size, sentence_size]
        query: [batch_size, sentence_size]
        """
        # query embedding
        u = torch.sum(self.B(query), dim=1)  # [batch_size, embedding_size]

        for _ in range(self.hops):
            # embed memory
            m = torch.sum(self.A(story), dim=2)  # [batch_size, memory_size, embedding_size]
            c = torch.sum(self.C(story), dim=2)

            # attention over memory
            p = self.softmax(torch.bmm(m, u.unsqueeze(2)).squeeze(2))  # [batch_size, memory_size]

            # output memory
            o = torch.bmm(p.unsqueeze(1), c).squeeze(1)  # [batch_size, embedding_size]

            # update u
            u = u + o  # hop mechanism

        return u  # 最终得到 memory 输出的向量
