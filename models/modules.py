import torch.nn as nn
import einops
import torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import to_2tuple, trunc_normal_
from local_attention import LocalAttention

class LocalWrapper(nn.Module):
    def __init__(self, input_dim, num_heads, window_size=7):
        super(LocalWrapper, self).__init__()
        self.local_attn = LocalAttention(dim=input_dim, window_size=window_size, shared_qk=True, causal=True)
        self.Wq = nn.Linear(input_dim, input_dim)
        self.Wk = nn.Linear(input_dim, input_dim)
        self.Wv = nn.Linear(input_dim, input_dim)

    def forward(self,x):
        b,h,w, c = x.shape
        x_flat = x.reshape(b,h*w,c)
        q = self.Wq(x_flat)
        k = self.Wk(x_flat)
        v = self.Wv(x_flat)
        out = self.local_attn(q,k,v)
        out = out.reshape(x.shape)
        return out


class SWA(nn.Module):
    def __init__(self, input_dim, num_heads, window_size=7):
        super(SWA, self).__init__()
        self.local_attn = LocalAttention(dim=input_dim, window_size=window_size, shared_qk=True, causal=True)
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.LN = nn.LayerNorm(input_dim)
        self.MLP = nn.Linear(input_dim, input_dim)
        self.act = nn.GELU()
        self.Wq = nn.Linear(input_dim, input_dim)
        self.Wk = nn.Linear(input_dim, input_dim)
        self.Wv = nn.Linear(input_dim, input_dim)
    def forward(self, x):
        """
        input: b,h,w,c
        out:   b,h,w,c
        """
       
        

        shift_size = int(self.window_size / 2)
        x_shifted = torch.roll(x, (shift_size,shift_size),(1,2))
        x_shifted = self.LN(x_shifted)
         #qkv issues
        x_shifted_flat = x_shifted.reshape(x_shifted.shape[0], x_shifted.shape[1]*x_shifted.shape[2], x_shifted.shape[3])
        q = self.Wq(x_shifted_flat)
        k = self.Wk(x_shifted_flat)
        v = self.Wv(x_shifted_flat)
        shifted_attn = self.local_attn(q,k,v)
        shifted_attn = shifted_attn.reshape(x.shape)
        x_unshifted = torch.roll(shifted_attn, (-shift_size,-shift_size),(1,2))
        x_unshifted_residual = x_unshifted + x
        x_unshifted = self.LN(x_unshifted_residual)
        x_unshifted_flat = x_unshifted.reshape(x_unshifted.shape[0], x_unshifted.shape[1]*x_unshifted.shape[2], x_unshifted.shape[3])
        x_unshifted = self.act(self.MLP(x_unshifted_flat)).reshape(x.shape) + x_unshifted_residual

        return x_unshifted


class MSA(nn.Module):
    def __init__(self, input_dim, query_seq_len,kv_seq_len,num_heads):
        super(MSA, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        
        self.query_seq_len = query_seq_len
        self.kv_seq_len = kv_seq_len
        self.head_dim = input_dim // num_heads
        
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        self.pe = nn.Parameter(torch.zeros(num_heads, query_seq_len, kv_seq_len) )
        self.output_linear = nn.Linear(input_dim, input_dim)
        
    def forward(self, queries, keys, values):
        """
            each shape: 
        """
        batch_size, seq_len, _ = queries.size()
        
        # queries = self.query(inputs)
        # keys = self.key(inputs)
        # values = self.value(inputs)
        
        queries = self.split_heads(queries)  # (batch_size, num_heads, seq_len, head_dim)
        keys = self.split_heads(keys)  # (batch_size, num_heads, seq_len, head_dim)
        values = self.split_heads(values)  # (batch_size, num_heads, seq_len, head_dim)
        
        attn_scores = torch.matmul(queries, keys.transpose(2, 3))  # (batch_size, num_heads, seq_len, seq_len)
        attn_scores = attn_scores #+ self.pe
        attn_probs = self.softmax(attn_scores)
        
        attended = torch.matmul(attn_probs, values)  
        
        attended = self.combine_heads(attended) 
        
        output = self.output_linear(attended)
        
        return output
    
    def split_heads(self, inputs):
        batch_size, seq_len, hidden_dim = inputs.size()
        inputs = inputs.view(batch_size, seq_len, self.num_heads, self.head_dim)
        inputs = inputs.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        return inputs
    
    def combine_heads(self, inputs):
        batch_size, _, seq_len, head_dim = inputs.size()
        inputs = inputs.transpose(1, 2)  # (batch_size, seq_len, num_heads, head_dim)
        inputs = inputs.contiguous().view(batch_size, seq_len, -1)  # (batch_size, seq_len, hidden_dim)
        return inputs


class Deformable(nn.Module):
    def __init__(self, input_dim, r, n_head, query_seq_len, s = 1):
        """
            input_dim: b,c,h,w
            output: b,h,w,c
        """
        super(Deformable, self).__init__()
        self.query_seq_len = query_seq_len
        self.kv_seq_len = int(query_seq_len/(r*r))
        self.Wq = nn.Linear(input_dim, input_dim)
        self.Wv = nn.Linear(input_dim, input_dim)
        self.Wk = nn.Linear(input_dim, input_dim)
        self.offset_network = OffsetNetwork(input_dim, r, kernel_size=7)
        self.r = r
        self.s = s
        self.softmax = nn.Softmax()
        self.LN = torch.nn.LayerNorm(input_dim)
        self.MSA = MSA(input_dim, num_heads=n_head, query_seq_len=self.query_seq_len,
                       kv_seq_len=self.kv_seq_len)
        self.MLP = nn.Linear(input_dim, input_dim)
        self.act = nn.GELU()


    def forward(self, x):
        """
            input_dim: b,c,h,w
        """

        # x_ln = self.LN(x.permute((0,2,3,1))).permute((0,3,1,2))
        x_ln = self.LN(x).permute((0,3,1,2))

        b,h,w,c = x.shape
        x_flat = x_ln.reshape(b, h*w, c)
        q =self.Wq(x_flat)
        offsets = self.offset_network(q.reshape(b,c,h,w)) 
        H_g, W_g = h/self.r, w/self.r
        #form ref pts
        ref_x = torch.arange(-1,1, 2/W_g) + 1/W_g
        ref_y = torch.arange(-1,1, 2/H_g) + 1/H_g
        grid_x, grid_y = torch.meshgrid(ref_x, ref_y, indexing="ij")
        refs = torch.stack([grid_x, grid_y], dim=-1).permute((2,0,1)).to(device=x_ln.device)

        #refs = torch.cat(tuple(torch.dstack([grid_x, grid_y])))

        deformed_pts = refs + self.s * F.tanh(offsets)
        deformed_pts = (deformed_pts + 1) * H_g / 2 #check if it is valid for w_g as well
        ccc=deformed_pts[(0, 0), ...]
        ccc = ccc.permute((0,2,3,1))
        interpolated = F.grid_sample(
            input= x.permute((0,3,1,2)),
            grid=deformed_pts[:, (1, 0), ...].permute((0,2,3,1)), # y, x -> x, y STM
            mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg
        
        interpolated = interpolated.reshape(interpolated.shape[0],
                                            interpolated.shape[2]*interpolated.shape[3],
                                             interpolated.shape[1] )
        v = self.Wv(interpolated)
        k = self.Wv(interpolated)

        z = self.MSA(q,k,v) + x.reshape(b, h*w, c)
        z_ln=self.LN(z)
        z_ln = self.act(self.MLP(z_ln))
        z_ln = z_ln + z
        z_ln = z_ln.reshape(b, w, h, c)
        return z_ln



        # .reshape(z_ln.shape[0], z_ln.shape[1]*z_ln.shape[2],z_ln.shape[3])
        #z_ln = self.act(self.MLP(z_ln)).reshape(z.shape)
        z = z_ln + z
        return z
        

class OffsetNetwork(nn.Module):
    def __init__(self, input_channel, stride, kernel_size):
        super(OffsetNetwork, self).__init__()
        self.dw_conv = nn.Conv2d(input_channel, input_channel, kernel_size, stride, kernel_size//2, groups=input_channel)
        self.gelu = nn.GELU()
        self.pointwise_conv = nn.Conv2d(input_channel, 2, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.gelu(x)
        x = self.pointwise_conv(x)
        return x


class Stage(nn.Module):
    def __init__(self, dim, spatial_dim, N, n_head, block_types, window_size=7):
        super(Stage, self).__init__()
        self.dim = dim
        self.N = N
        self.n_head = n_head
        self.window_size = window_size
        self.blocks = nn.ModuleList()
        for type in block_types:
            if type == "Local":
                self.blocks.append(LocalWrapper(input_dim=dim, num_heads=n_head, window_size=window_size))
            elif type == "Shifted":
                self.blocks.append(SWA(input_dim=dim, num_heads=n_head, window_size=window_size))
            elif type == "Deformable":
                self.blocks.append(Deformable(input_dim=dim, r=2, n_head=n_head, query_seq_len=spatial_dim*spatial_dim))
    def forward(self,x):
        for block in self.blocks:
            x = block(x)
        return x
    
            


    

def main():

    model = Deformable(input_dim=96, query_seq_len=32*32, r=8, n_head=3)
    x = torch.zeros(5,32,32,96)
    model(x)
    b,c,h,w = x.shape


    
    model = LocalAttention(dim=96, window_size=7, shared_qk=True, causal=True)
    q = torch.zeros(5,56,56,96)
    out = model(q,q,q)
    
    model = SWA(input_dim=96, num_heads=3, window_size=7)
    x = torch.zeros(5,56,56,96)
    out = model(x)
    44

if __name__ == "__main__":
    main()

