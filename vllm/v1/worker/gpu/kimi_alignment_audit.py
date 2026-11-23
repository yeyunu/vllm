"""Capture opt-in Kimi serving boundaries from the real model runner."""
import json
import os
import time
from pathlib import Path
import torch

def cpu_tensor(value):
    if not isinstance(value,torch.Tensor): return None
    if getattr(value,'is_nested',False):
        return {'values':value.values().detach().cpu(), 'offsets':value.offsets().detach().cpu()}
    return value.detach().cpu()


def capture_vllm_v2_step(runner, logits, selected_hidden, batch):
    """Capture eight real V2 sampling inputs before any logits processing."""
    root=os.environ.get('KIMI_ALIGNMENT_AUDIT_DIR')
    if not root: return
    req_ids=list(batch.req_ids)
    if not req_ids or any('warmup' in str(x) for x in req_ids): return
    if os.environ.get("KIMI_ALIGNMENT_SERVING_PROBE") == "1" and any(str(r).startswith("alignment-probe-") for r in req_ids):
        from vllm.v1.engine.kimi_alignment_probe import capture_probe_logits
        from vllm.distributed import get_tensor_model_parallel_rank
        capture_probe_logits(logits,batch,get_tensor_model_parallel_rank())
    count=getattr(runner,'_alignment_capture_count',0)
    if count >= 8: return
    runner._alignment_capture_count=count+1
    from vllm.distributed import get_tensor_model_parallel_rank
    rank=get_tensor_model_parallel_rank()
    from vllm_ascend.ops.kimi_alignment.kda_trace import save_runner_traces
    save_runner_traces(runner,batch,rank,count)
    if rank != 0: return
    requests=[]
    for req_id in req_ids:
        idx=runner.req_states.req_id_to_index[req_id]
        prompt_len=int(runner.req_states.prompt_len.np[idx])
        requests.append({'req_id':req_id,'prompt_length':prompt_len,
            'prompt_token_ids':cpu_tensor(runner.req_states.all_token_ids.gpu[idx,:prompt_len])})
    record={'rank':rank,'step':count,'requests':requests,
            'logits':cpu_tensor(logits),'selected_hidden':cpu_tensor(selected_hidden)}
    language_model=runner.model.get_language_model()
    record['post_norm_hidden']=cpu_tensor(getattr(language_model,'_alignment_post_norm_hidden',None))
    for key in ('input_ids','positions','logits_indices','query_start_loc'):
        record[key]=cpu_tensor(getattr(batch,key,None))
    for key in ('num_computed_tokens_np','query_start_loc_np','is_prefilling_np'):
        value=getattr(batch,key,None)
        if value is not None:record[key]=value.tolist()
    path=Path(root)/'vllm_steps';path.mkdir(parents=True,exist_ok=True)
    torch.save(record,path/(str(time.time_ns())+'_rank'+str(rank)+'.pt'))
    print('KIMI_ALIGNMENT_VLLM_STEP '+json.dumps({'step':count,'rank':rank,'requests':len(requests)}),flush=True)
