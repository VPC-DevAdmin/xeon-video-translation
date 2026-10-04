import torch,json
from wan.modules.attention import flash_attention
q=torch.randn(2,13,4,64,device='cuda',dtype=torch.bfloat16);k=torch.randn(2,17,4,64,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
ql=torch.tensor([13,9],device='cuda');kl=torch.tensor([12,17],device='cuda')
y=flash_attention(q,k,v,q_lens=ql,k_lens=kl)
mask=(torch.arange(17,device='cuda')[None,:]<kl[:,None])[:,None,None,:]
z=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),attn_mask=mask).transpose(1,2)
z[1,9:]=0
error=(y-z).abs().max().item()
assert error<0.025,error
assert torch.count_nonzero(y[1,9:])==0
print(json.dumps({'variable_length_attention_max_error_bf16':error,'padding_preserved':True}),flush=True)
