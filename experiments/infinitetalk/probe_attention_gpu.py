import torch,time,json
q=torch.randn(1,40,8192,128,device='cuda',dtype=torch.bfloat16)
for label,mask in [('unmasked',None),('key-padding',torch.ones(1,1,1,8192,device='cuda',dtype=torch.bool))]:
 torch.cuda.synchronize();t=time.perf_counter()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
  y=torch.nn.functional.scaled_dot_product_attention(q,q,q,attn_mask=mask)
 torch.cuda.synchronize()
 print(json.dumps({'label':label,'seconds':time.perf_counter()-t,'ops':[e.key for e in p.key_averages() if 'attention' in e.key]}),flush=True)
