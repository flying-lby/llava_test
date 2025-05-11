import torch
import math
from torch import nn

class Multi_atten(nn.Module):
    def __init__(self, embed_size, q_k_size, v_size, head):
        super(Multi_atten,self).__init__()
        
        self.head = head
        self.q_k_size = q_k_size
        self.v_size = v_size
        self.embed_size = embed_size
        
        self.WK = nn.Linear(embed_size, q_k_size*head)
        self.Wq = nn.Linear(embed_size, q_k_size*head)
        self.Wv = nn.Linear(embed_size, v_size*head)
        
        self.softmax = nn.Softmax(dim = -1)
        self.fc_out = nn.Linear(head * v_size, )
        
    def forward(self,x_q,x_k_v,mask):
        # batchsize seq_length hidden_dim
        batch_size = x_q.shape[0]
        q = self.Wq(x_q) 
        k = self.WK(x_k_v)
        v = self.Wv(x_k_v)
        
        # b head seq head_dim
        q = q.reshape(batch_size, -1, self.head, self.q_k_size).transpose(1,2)
        k = k.reshape(batch_size, -1, self.head, self.q_k_size).transpose(1,2)
        v = v.reshape(batch_size, -1, self.head, self.v_size).transpose(1,2)
        # b head head_dim seq 
        k_t = k.transpose(-2, -1)
        d_k = self.q_k_size
        q_k = self.softmax(torch.matmul(q, k_t)/math.sqrt(d_k))
        
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.head, -1, -1)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = self.softmax(attention_scores)
        
        attention = torch.matmul(attention_weights, v).transpose(1,2).contiguous()  # (batch_size, q_seq_len, head, v_size)
        attention = attention.view(batch_size, -1, self.head * self.v_size) # (batch_size, q_seq_len, head * v_size)
        
        output = self.fc_out(attention)  # (batch_size, q_seq_len, embed_size)

        
        return output
        
        

# import torch
# import torch.nn as nn
# import math

# class MultiHeadAttention(nn.Module):
#     def __init__(self, embed_size, q_k_size, v_size, head):
#         super(MultiHeadAttention, self).__init__()
#         self.head = head
#         self.q_k_size = q_k_size
#         self.v_size = v_size
#         self.embed_size = embed_size

#         # 线性变换矩阵
#         self.Wq = nn.Linear(embed_size, q_k_size * head)
#         self.Wk = nn.Linear(embed_size, q_k_size * head)
#         self.Wv = nn.Linear(embed_size, v_size * head)

#         self.softmax = nn.Softmax(dim=-1)
#         self.fc_out = nn.Linear(head * v_size, embed_size)  # 最终输出变换

#     def forward(self, x_q, x_k_v, mask=None):
#         batch_size = x_q.shape[0]

#         # 计算 Q, K, V
#         Q = self.Wq(x_q)  # (batch_size, q_seq_len, head * q_k_size)
#         K = self.Wk(x_k_v)  
#         V = self.Wv(x_k_v)  

#         # 变形 -> (batch_size, head, seq_len, q_k_size/v_size)
#         Q = Q.view(batch_size, -1, self.head, self.q_k_size).transpose(1, 2)
#         K = K.view(batch_size, -1, self.head, self.q_k_size).transpose(1, 2)
#         V = V.view(batch_size, -1, self.head, self.v_size).transpose(1, 2)

#         # 计算注意力分数
#         K_t = K.transpose(-2, -1)  # 转置 K (batch_size, head, q_k_size, seq_len)
#         d_k = self.q_k_size  # 计算 d_k
#         attention_scores = torch.matmul(Q, K_t) / math.sqrt(d_k)  # (batch_size, head, q_seq_len, k_seq_len)

#         # 处理 Mask
#         if mask is not None:
#             mask = mask.unsqueeze(1).expand(-1, self.head, -1, -1)  # (batch_size, head, seq_q, seq_k)
#             attention_scores = attention_scores.masked_fill(mask == 0, -1e9)  # 负无穷，防止 softmax 影响

#         # 计算 softmax 权重
#         attention_weights = self.softmax(attention_scores)  

#         # 计算加权 V
#         attention = torch.matmul(attention_weights, V)  # (batch_size, head, q_seq_len, v_size)

#         # 变形回原始尺寸
#         attention = attention.transpose(1, 2).contiguous()  # (batch_size, q_seq_len, head, v_size)
#         attention = attention.view(batch_size, -1, self.head * self.v_size)  # (batch_size, q_seq_len, head * v_size)

#         # 通过最终的线性变换
#         output = self.fc_out(attention)  # (batch_size, q_seq_len, embed_size)

#         return output