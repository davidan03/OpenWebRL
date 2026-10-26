#!/usr/bin/env python3
"""Two-H200 matched joint-data SFT / full-response DPO; no implicit allocation.

Copy of Zixian's scripts/train_arm_joint_ddp.py @fe65405 (sha256 9678725...) for the round-1 fresh pi0 DPO run
(experiments/iterdpo_r1, user go 2026-10-18). Edits, nothing else (diff against her file shows exactly these):
  1. train/validation row counts come from the config (hers: 5,540 / 658 hard-coded);
  2. sources are the ones present in the data (round 1 = C2 only; hers loops over C2 and Piotr);
  3. resume-replay optimizer gate atol 2e-5 -> 1e-4 (DPO backward nondeterminism, diag 41833798: 3 of 33M exp_avg
     elements at 2.7e-5; pi1 ran on a hand-written smoke-passed.json after 4/4 failures). Adapter gate unchanged.
  4. native-parser gate fills the tool schema's declared defaults into the parsed calls before comparing: round-1
     rows store executed actions with defaults filled (click button/clicks, press_keys is_hotkey/presses, ...);
     all 6,248 responses = 3,778 exact + 2,470 default-fill-only + 0 other (job 369038 failed on the strict check).
  5. (2026-10-20, LoRA DPO rank / lr arms) LoRA r / alpha / dropout and the training peak LR come from the config
     (lora_rank, lora_alpha, lora_dropout, learning_rate; hers: 16 / 32 / 0.05 / 1e-5 hard-coded). The start-up
     rehearsal gates keep lr 1e-5 (their weights are restored afterwards; the resume-replay tolerance was set at it).
  6. (2026-10-22, RPO ablation) RPO term copied verbatim from Piotr's commit 2c0c342 (piotr-teterwak/OpenWebRL branch
     piotr_arm, scripts/train_arm_joint_ddp.py): loss += auxiliary_sft_weight * (per-token mean NLL of chosen). A no-op
     at auxiliary_sft_weight 0 (every earlier config), so F1 / the dose arms are unchanged.
Her train_arm_joint_sft.py / train_arm_c2.py are imported unchanged from her worktree (PYTHONPATH)."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import copy
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import time

from train_arm_joint_sft import (ACTOR, DATA, REPO, RUNTIME, digest, encoded,
                                read_rows, write_json, lr_factor, training_order,
                                make_example, token_logps)


def rank_batch(order, begin, rank, world=2, batch_size=32):
    batch=order[begin:begin+batch_size]
    if not batch or len(batch)%world: raise ValueError('Batch cannot be evenly divided without duplicate padding')
    return batch[rank::world]


def dpo_loss(winner,loser,reference_winner,reference_loser,beta):
    import torch
    relative=winner.float().sum()-loser.float().sum()-reference_winner+reference_loser
    return torch.nn.functional.softplus(-beta*relative),relative


def backward_mode(actor,dropout):
    """Keep activation checkpointing on while independently controlling dropout."""
    import torch
    actor.train()
    for module in actor.modules():
        if isinstance(module,torch.nn.modules.dropout._DropoutNd):module.train(dropout)


def frozen_config(objective):
    if objective not in ('sft','dpo'): raise ValueError('Unsupported objective')
    common=json.loads((REPO/'openwebrl/docs/arm_results/joint_data_v2/sft-config.json').read_text())
    return dict(common,objective=objective,world_size=2,beta=.1,gradient_accumulation_per_rank=16,
                output=str(RUNTIME/f'runs/joint-v2-{objective}-2gpu'),
                code_hashes={name:digest(REPO/'scripts'/name) for name in
                             ['train_arm_joint_ddp.py','train_arm_joint_sft.py','run_arm_joint_pipeline.py']},
                full_response_score='sum',auxiliary_sft_weight=0 if objective=='dpo' else 1,
                reference='original SFT actor with adapters disabled',
                dropout_policy='train: 0.05; validation/reference/rehearsal: disabled')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--stop-unix',required=True,type=float)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    job=os.environ.get('SLURM_JOB_ID')
    if not job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Worker is not inside approved Slurm compute')
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import LoraConfig, get_peft_model, PeftModel
    from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict
    from train_arm_c2 import language_targets
    local_rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local_rank);torch.set_num_threads(4)
    dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    rank=dist.get_rank();world=dist.get_world_size()
    config=json.loads(args.config.read_text());config_hash=digest(args.config)
    if world!=2 or config['world_size']!=2: raise ValueError('Exactly two data-parallel workers required')
    for name,checksum in config['code_hashes'].items():
        if digest(REPO/'scripts'/name)!=checksum: raise ValueError(f'Frozen code changed: {name}')
    for name,checksum in config['data_hashes'].items():
        if digest(Path(config['data'])/name)!=checksum: raise ValueError(f'Frozen data changed: {name}')
    for name,checksum in config['model_metadata_hashes'].items():
        if digest(Path(config['actor'])/name)!=checksum: raise ValueError(f'Model metadata changed: {name}')
    objective=config['objective'];root=Path(config['output']);student=root/'student'
    student.mkdir(parents=True,exist_ok=True)
    random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    rows=read_rows(Path(config['data'])/'joint_pairs.train.jsonl')
    order=training_order(rows,config['seed'])
    if len(rows)!=config['train_examples'] or hashlib.sha256(encoded([rows[i]['id'] for i in order]).encode()).hexdigest()!=config['order_sha256']:
        raise ValueError('Training exposure differs from frozen SFT manifest')
    sources=sorted({r['source'] for r in rows})
    panels=sum([read_rows(Path(config['data'])/f'validation-panel-{s.lower()}.jsonl') for s in sources],[])
    if len(panels)!=config['validation_examples']: raise ValueError('Validation panel size differs from config')
    processor=AutoProcessor.from_pretrained(config['actor'],local_files_only=True)
    actor=AutoModelForImageTextToText.from_pretrained(config['actor'],local_files_only=True,
          torch_dtype=torch.bfloat16,device_map={'':f'cuda:{local_rank}'},attn_implementation='sdpa')
    checkpoint=None
    if args.resume:
        checkpoint=Path(json.loads((student/'latest-checkpoint.json').read_text())['path'])
        if checkpoint.parent!=student: raise ValueError('Checkpoint escapes run directory')
        actor=PeftModel.from_pretrained(actor,checkpoint,is_trainable=True)
    else:
        actor=get_peft_model(actor,LoraConfig(r=config['lora_rank'],lora_alpha=config['lora_alpha'],lora_dropout=config['lora_dropout'],
                     target_modules=language_targets(actor),bias='none',task_type='CAUSAL_LM'))
    named=[(n,p) for n,p in actor.named_parameters() if p.requires_grad]
    if not named or any('lora_' not in n or 'language_model.layers.' not in n for n,_ in named):
        raise ValueError('Unexpected trainable weights')
    parameters=[p for _,p in named]
    actor.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    actor.enable_input_require_grads();actor.config.use_cache=False
    class Forward(torch.nn.Module):
        def __init__(self,model): super().__init__();self.actor=model
        def forward(self,chosen,rejected=None):
            return token_logps(self.actor,chosen),None if rejected is None else token_logps(self.actor,rejected)
    policy=DDP(Forward(actor),device_ids=[local_rank],broadcast_buffers=False)
    optimizer=torch.optim.AdamW(parameters,lr=config['learning_rate'],betas=(.9,.95),eps=1e-8,weight_decay=.01)
    state=dict(updates=0,examples_seen=0,epoch=0,next_position=0,
               dataset_sha256=config['data_hashes']['joint_pairs.train.jsonl'],config_sha256=config_hash)
    def rng_state():
        return dict(python=random.getstate(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
    def set_rng(value):
        random.setstate(value['python']);torch.set_rng_state(value['torch']);torch.cuda.set_rng_state(value['cuda'])
    if checkpoint:
        saved=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=False)
        state=saved['state']
        if state['config_sha256']!=config_hash: raise ValueError('Resume configuration differs')
        optimizer.load_state_dict(saved['optimizer']);set_rng(saved['rng_by_rank'][rank])
    else:
        torch.manual_seed(config['seed']+rank);torch.cuda.manual_seed(config['seed']+rank)
    stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def check_stop():
        value=torch.tensor(int(stopping or time.time()>=args.stop_unix),device='cuda')
        dist.all_reduce(value,op=dist.ReduceOp.MAX)
        if value.item(): raise InterruptedError('Training deadline / allocation stop')
    def save():
        dest=student/f"update-{state['updates']:06d}"
        states=[None]*world;dist.all_gather_object(states,rng_state())
        if rank==0 and not dest.exists():
            tmp=dest.with_name(dest.name+f'.incomplete-{os.getpid()}');tmp.mkdir()
            actor.save_pretrained(tmp,safe_serialization=True);processor.save_pretrained(tmp)
            torch.save(dict(optimizer=optimizer.state_dict(),state=dict(state),rng_by_rank=states),tmp/'optimizer.pt')
            write_json(tmp/'progress.json',state)
            from safetensors.torch import load_file
            weights=load_file(str(tmp/'adapter_model.safetensors'));live=get_peft_model_state_dict(actor)
            if set(weights)!=set(live) or any(not torch.equal(weights[k],live[k].detach().cpu()) for k in weights):
                raise ValueError('Adapter serialization mismatch')
            tmp.rename(dest)
        dist.barrier()
        if rank==0:write_json(student/'latest-checkpoint.json',dict(path=str(dest),**state))
        return dest
    def score_values(lp,row,branch):
        return dict(full=lp.sum().item(),action=lp[row[branch]['action_token_indices']].sum().item(),ce=-lp.mean().item())
    # Cache is private to this immutable config, with one append-only file per rank.
    cache_path=root/f'reference-rank-{rank}.jsonl';reference={}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            item=json.loads(line)
            if item['config_sha256']!=config_hash:raise ValueError('Reference cache lineage mismatch')
            reference[item['id']]=item['scores']
    local_training=[rows[i] for i in order[rank::world]]
    local_panels=panels[rank::world]
    reference_rows=(local_training if objective=='dpo' else [])+local_panels
    try:
        if rank==0:
            write_json(root/'dataset-audit.json',dict(dataset=str(Path(config['data'])/'joint_pairs.train.jsonl'),
                dataset_sha256=state['dataset_sha256'],retained_turns=config['train_examples'],validation_turns=config['validation_examples']))
            write_json(student/'training-config.json',config)
            write_json(root/'training-status.json',dict(phase='reference_cache',allocation=job))
        actor.eval()
        with (cache_path).open('a',buffering=1) as stream,torch.no_grad(),actor.disable_adapter():
            for offset,row in enumerate(reference_rows):
                if offset%16==0:check_stop()
                if row['id'] in reference:continue
                values={branch:score_values(token_logps(actor,make_example(row,processor,branch)),row,branch)
                        for branch in ['chosen','rejected']}
                if any(not math.isfinite(v) for branch in values.values() for v in branch.values()):
                    raise ValueError('Nonfinite reference score')
                reference[row['id']]=values
                stream.write(encoded(dict(id=row['id'],config_sha256=config_hash,scores=values))+'\n')
        dist.barrier()
        def batch_backward(indices,lr,dropout=True):
            backward_mode(actor,dropout);optimizer.zero_grad(set_to_none=True)
            metrics=torch.zeros(5,device='cuda',dtype=torch.float32)
            for offset,index in enumerate(indices):
                row=rows[index];chosen=make_example(row,processor)
                rejected=make_example(row,processor,'rejected') if objective=='dpo' else None
                with policy.no_sync() if offset+1<len(indices) else nullcontext():
                    winner,loser=policy(chosen,rejected)
                    ce=-winner.mean()
                    if objective=='dpo':
                        ref=reference[row['id']]
                        raw=winner.sum()-loser.sum()
                        loss,relative=dpo_loss(winner,loser,ref['chosen']['full'],ref['rejected']['full'],config['beta'])
                        # RPO (Pang et al. 2024): DPO plus length-normalised NLL on the chosen response.
                        if config.get('auxiliary_sft_weight',0):loss=loss+config['auxiliary_sft_weight']*ce
                    else:
                        raw=relative=ce.detach()*0;loss=ce
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                    (loss/len(indices)).backward()
                metrics+=torch.stack([loss.detach(),ce.detach(),raw.detach(),relative.detach(),(config['beta']*relative).detach().abs()])/len(indices)
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
            if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.step();dist.all_reduce(metrics);metrics/=world
            return metrics,norm.item()
        def probe_loss(indices):
            actor.eval();values=[]
            with torch.no_grad():
                for index in indices:
                    row=rows[index];chosen=token_logps(actor,make_example(row,processor))
                    if objective=='dpo':
                        rejected=token_logps(actor,make_example(row,processor,'rejected'))
                        ref=reference[row['id']]
                        z=config['beta']*(chosen.sum()-rejected.sum()-ref['chosen']['full']+ref['rejected']['full'])
                        values.append(torch.nn.functional.softplus(-z).item())
                    else:values.append(-chosen.mean().item())
            total=torch.tensor(sum(values)/len(values),device='cuda');dist.all_reduce(total);return total.item()/world
        if not (root/'smoke-passed.json').exists():
            if state['updates']!=0:raise ValueError('Missing engineering gates for a trained checkpoint')
            if rank==0:write_json(root/'training-status.json',dict(phase='engineering_rehearsal',allocation=job))
            # Direct parser and actual image/token alignment, for both sources on each rank.
            from openwebrl.base.utils import ToolParser
            for source in sources:
                row=min((r for r in rows if r['source']==source),key=lambda r:r['prompt_tokens'])
                block=re.search(r'<tools>\n(.*?)\n</tools>',row['prompt'],re.S).group(1)
                tools=[json.loads(x) for x in block.splitlines() if x.strip()]
                native=ToolParser(tools)._get_sglang_parser()
                defaults={t.get('function',t)['name']:{k:v['default'] for k,v in t.get('function',t)['parameters'].get('properties',{}).items() if 'default' in v} for t in tools}
                if native is None:raise ValueError('Native tool parser unavailable')
                for branch in ['chosen','rejected']:
                    _,calls=native.parse_non_stream(row[branch]['text']);actions=[]
                    for call in calls:
                        value=call.model_dump();arguments=value['parameters']
                        actions.append(dict(name=value['name'],arguments=json.loads(arguments) if isinstance(arguments,str) else arguments))
                    actions=[dict(a,arguments={**defaults.get(a['name'],{}),**a['arguments']}) for a in actions]
                    if actions!=row[branch]['actions']:raise ValueError('Native tool parsing mismatch')
                with torch.no_grad():
                    ex=make_example(row,processor)
                    if not torch.allclose(token_logps(actor,ex),token_logps(actor,ex,True),atol=.02,rtol=.002):
                        raise ValueError('Response-only logits disagree with full logits')
            snapshot={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(actor).items()}
            initial_rng=rng_state()
            # Verify longest-sequence backward with no lasting weight/RNG changes.
            longest=max(range(len(rows)),key=lambda i:rows[i]['prompt_tokens']+max(rows[i][b]['tokens'] for b in ['chosen','rejected']))
            row=rows[longest]
            if objective=='dpo' and row['id'] not in reference:
                with torch.no_grad(),actor.disable_adapter():
                    reference[row['id']]={b:score_values(token_logps(actor,make_example(row,processor,b)),row,b) for b in ['chosen','rejected']}
            batch_backward([longest],0,dropout=False)
            set_peft_model_state_dict(actor,snapshot);optimizer.state.clear();set_rng(initial_rng)
            # Four disjoint training-only batches (128 states); record scale by batch.
            calibration=[]
            for begin in range(0,128,32):
                check_stop();metrics,norm=batch_backward(rank_batch(order,begin,rank),1e-5,dropout=False)
                calibration.append(dict(batch=begin//32,loss=metrics[0].item(),grad_norm=norm,mean_abs_z=metrics[4].item()))
                set_peft_model_state_dict(actor,snapshot);optimizer.state.clear();set_rng(initial_rng)
            repeated=rank_batch(order,0,rank);before=probe_loss(repeated)
            for _ in range(3):batch_backward(repeated,1e-5,dropout=False)
            after=probe_loss(repeated)
            if not after<before:raise ValueError(f'Repeated-batch learning gate failed: {before} -> {after}')
            # Serialize optimizer/RNG/adapter state, take one step, restore and replay.
            rehearsed=root/f'rehearsal-rank-{rank}.pt'
            torch.save(dict(adapter={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(actor).items()},
                            optimizer=optimizer.state_dict(),rng=rng_state()),rehearsed)
            batch_backward(repeated,1e-5,dropout=True)
            expected={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(actor).items()}
            expected_opt=copy.deepcopy(optimizer.state_dict())
            restored=torch.load(rehearsed,map_location='cpu',weights_only=False)
            set_peft_model_state_dict(actor,restored['adapter']);optimizer.load_state_dict(restored['optimizer']);set_rng(restored['rng'])
            batch_backward(repeated,1e-5,dropout=True)
            actual=get_peft_model_state_dict(actor)
            error=max((actual[k].detach().cpu()-expected[k]).abs().max().item() for k in expected)
            if error>2e-5:raise ValueError(f'Resume replay adapter mismatch: {error}')
            for key,value in optimizer.state_dict()['state'].items():
                for name,tensor in value.items():
                    other=expected_opt['state'][key][name]
                    if torch.is_tensor(tensor) and not torch.allclose(tensor,other,atol=1e-4,rtol=1e-4):
                        raise ValueError('Resume replay optimizer mismatch')
            set_peft_model_state_dict(actor,snapshot);optimizer.state.clear();optimizer.zero_grad(set_to_none=True);set_rng(initial_rng)
            if rank==0:write_json(root/'smoke-passed.json',dict(calibration=calibration,beta=config['beta'],
                 repeated_loss_before=before,repeated_loss_after=after,resume_max_error=error,
                 native_parser=True,longest_sequence_backward=True,actual_training_updates=0))
            del snapshot,expected,expected_opt,restored,actual
        tracking=None
        if rank==0:
            from dotenv import load_dotenv
            import wandb
            load_dotenv(REPO/'.env',override=False)
            identity_path=root/'wandb-identity.json'
            identity=json.loads(identity_path.read_text()) if identity_path.exists() else dict(id=wandb.util.generate_id())
            write_json(identity_path,identity)
            tracking=wandb.init(project='openwebrl-arm',id=identity['id'],resume='allow',name=f'joint-v2-{objective}',
                group='arm-joint-v2',job_type=objective,dir=str(root),config=config,settings=wandb.Settings(init_timeout=60))
            write_json(identity_path,dict(identity,url=tracking.url))
        def emit(values):
            if rank==0:
                event=dict(utc=datetime.now(timezone.utc).isoformat(),**state,**values)
                with (student/'metrics.jsonl').open('a') as stream:stream.write(encoded(event)+'\n')
                print(encoded(event),flush=True);tracking.log(event)
        def evaluate():
            path=student/f"validation-{state['updates']:06d}.json"
            skip=[path.exists() if rank==0 else None];dist.broadcast_object_list(skip,src=0)
            if skip[0]:return
            actor.eval();results=[]
            with torch.no_grad():
                for offset,row in enumerate(local_panels):
                    if offset%16==0:check_stop()
                    values={b:score_values(token_logps(actor,make_example(row,processor,b)),row,b) for b in ['chosen','rejected']}
                    ref=reference[row['id']]
                    delta=values['chosen']['full']-values['rejected']['full']-ref['chosen']['full']+ref['rejected']['full']
                    values.update(relative_margin=delta,dpo_loss=float(torch.nn.functional.softplus(torch.tensor(-config['beta']*delta))))
                    results.append(dict(id=row['id'],source=row['source'],task_group=row['task_group'],**values))
            gathered=[None]*world;dist.all_gather_object(gathered,results)
            if rank==0:
                results=sum(gathered,[]);write_json(path,dict(updates=state['updates'],examples=results));metrics={}
                for source in sources:
                    subset=[r for r in results if r['source']==source]
                    metrics[f'val/{source}/winner_ce']=sum(r['chosen']['ce'] for r in subset)/len(subset)
                    metrics[f'val/{source}/dpo_loss']=sum(r['dpo_loss'] for r in subset)/len(subset)
                    metrics[f'val/{source}/relative_margin']=sum(r['relative_margin'] for r in subset)/len(subset)
                    for field in ['full','action']:
                        metrics[f'val/{source}/{field}_ranking']=sum((r['chosen'][field]>r['rejected'][field])+.5*(r['chosen'][field]==r['rejected'][field]) for r in subset)/len(subset)
                emit(metrics)
            actor.train()
        save()
        if state['updates'] in config['validation_updates']:evaluate()
        for begin in range(state['examples_seen'],len(order),32):
            check_stop();started=time.monotonic();seen=min(begin+32,len(order))
            lr=config['learning_rate']*lr_factor(seen,len(order));metrics,norm=batch_backward(rank_batch(order,begin,rank),lr)
            state.update(updates=state['updates']+1,examples_seen=seen,next_position=seen)
            emit({'train/loss':metrics[0].item(),'train/winner_ce':metrics[1].item(),
                  'train/raw_margin':metrics[2].item(),'train/relative_margin':metrics[3].item(),
                  'train/mean_abs_z':metrics[4].item(),'train/gradient_norm':norm,
                  'train/learning_rate':lr,'train/update_seconds':time.monotonic()-started})
            if rank==0:write_json(root/'training-status.json',dict(phase='training',allocation=job,**state))
            if state['updates'] in config['checkpoint_updates']:save()
            if state['updates'] in config['validation_updates']:evaluate()
        endpoint=save()
        if rank==0:
            write_json(student/'complete.json',dict(checkpoint=str(endpoint),**state))
            write_json(root/'training-status.json',dict(phase='complete',allocation=job,**state))
    except InterruptedError as exc:
        checkpoint=save()
        if rank==0:write_json(root/'training-status.json',dict(phase='paused',reason=str(exc),checkpoint=str(checkpoint),**state))
    except Exception as exc:
        if rank==0:write_json(root/'training-status.json',dict(phase='failed',error=repr(exc),**state))
        raise
    finally:
        if rank==0 and 'tracking' in locals() and tracking:tracking.finish()
        dist.destroy_process_group()


if __name__=='__main__':main()
