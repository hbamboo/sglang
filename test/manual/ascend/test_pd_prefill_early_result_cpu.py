"""CPU control-flow contracts; no torch, SG imports, NPU or HTTP.

The new helper is imported directly. Existing loop/copy method bodies are
extracted from the candidate source and executed with explicit CPU fakes; this
checks ordering/markers, not TorchNPU event readiness or model correctness.
"""
import ast
from collections import deque
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT/'python/sglang/srt'
spec = importlib.util.spec_from_file_location('early_result_under_test', SRT/'disaggregation/early_prefill_result.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def method(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    fn.decorator_list = []
    future = ast.parse('from __future__ import annotations').body[0]
    mod = ast.fix_missing_locations(ast.Module(body=[future, fn], type_ignores=[]))
    exec(compile(mod, str(path), 'exec'), namespace)
    return namespace[method_name]


class Event:
    def __init__(self, ready=True, error=False):
        self.ready, self.error, self.queries = ready, error, 0

    def query(self):
        self.queries += 1
        if self.error:
            raise RuntimeError('query failure')
        return self.ready


def request(row):
    return NS(kv=NS(req_pool_idx=row), origin_input_ids=[1, 2], extend_range=NS(end=2),
              inflight_middle_chunks=0, pending_bootstrap=False, finished_reason=None,
              to_finish=None, grammar=None, session=None, multimodal_inputs=None,
              metadata_buffer_index=row)


def batch(name, row):
    b = NS(name=name, forward_mode=NS(name='EXTEND'), is_dllm=lambda:False,
           chunked_req=None, spec_info=None, return_logprob=False, reqs=[request(row)])
    b.copy = lambda:b
    return b


def result(event=None):
    return NS(copy_done_recorded=True, copy_done=event or Event(), delay_sample_func=None,
              has_sampled_token_ids=True, next_token_ids=NS(device=NS(type='cpu')),
              next_draft_input=None, can_run_cuda_graph=False, routed_experts_output=None,
              indexer_topk_output=None, expert_distribution_metrics=None,
              auxiliary_host_output=None)


def scheduler(previous=None, previous_result=None):
    cache_cls = type('RadixCache', (), {'__module__':'sglang.srt.mem_cache.radix_cache'})
    s = NS(ps=NS(**{k:1 for k in ('tp_size','pp_size','dp_size','attn_tp_size','attn_cp_size',
                                 'attn_dcp_size','attn_dp_size','moe_ep_size','moe_dp_size')}),
           enable_overlap=True, spec_algorithm=NS(is_none=lambda:True), tree_cache=cache_cls(),
           last_batch=previous, chunked_req=None, result_queue=deque(), log=[])
    s.tree_cache.disable = False
    for key in ('enable_pdmux','enable_staging','enable_unified_memory','enable_hierarchical_cache',
                'enable_session_radix_cache','enable_hicache_storage','enable_hisparse','enable_dp_attention'):
        setattr(s, key, False)
    if previous is not None:
        s.result_queue.append((previous, previous_result))
    s.process_batch_result = lambda b,r:s.log.append(('process',b.name))
    return s


class Contracts(unittest.TestCase):
    def fixture(self, ready=True):
        a,b = batch('A',1),batch('B',2)
        r = result(Event(ready))
        return scheduler(a,r),a,b,r

    def test_ready_processes_queue_head_once(self):
        s,a,b,r = self.fixture()
        before_req_rows = [a.reqs[0].kv.req_pool_idx,b.reqs[0].kv.req_pool_idx]
        self.assertTrue(helper.try_process_ready_prefill_result(s,b))
        self.assertEqual(s.log,[('process','A')]);self.assertEqual(len(s.result_queue),0)
        self.assertIs(s.last_batch,a);self.assertEqual(r.copy_done.queries,1)
        self.assertEqual(before_req_rows,[a.reqs[0].kv.req_pool_idx,b.reqs[0].kv.req_pool_idx])

    def test_not_ready_preserves_state(self):
        s,a,b,r = self.fixture(False)
        self.assertFalse(helper.try_process_ready_prefill_result(s,b))
        self.assertEqual(s.log,[]);self.assertIs(s.result_queue[0][0],a)
        self.assertIs(s.result_queue[0][1],r);self.assertEqual(r.copy_done.queries,1)

    def test_unrecorded_and_incomplete_copy_do_not_query(self):
        for field,value in [('copy_done_recorded',False),('delay_sample_func',lambda:None),
                            ('has_sampled_token_ids',False),('next_draft_input',object()),
                            ('can_run_cuda_graph',True),('routed_experts_output',object()),
                            ('indexer_topk_output',object()),('expert_distribution_metrics',object()),
                            ('auxiliary_host_output',object())]:
            with self.subTest(field=field):
                s,a,b,r=self.fixture();event=r.copy_done;setattr(r,field,value)
                self.assertFalse(helper.try_process_ready_prefill_result(s,b))
                self.assertEqual(event.queries,0);self.assertIs(s.result_queue[0][1],r)
        s,a,b,r=self.fixture();r.copy_done=None
        self.assertFalse(helper.try_process_ready_prefill_result(s,b))
        s,a,b,r=self.fixture();r.next_token_ids.device.type='npu'
        self.assertFalse(helper.try_process_ready_prefill_result(s,b));self.assertEqual(r.copy_done.queries,0)

    def test_shared_request_or_pool_row_rejected(self):
        for same_object in (False,True):
            s,a,b,r=self.fixture()
            if same_object:b.reqs=a.reqs
            else:b.reqs[0].kv.req_pool_idx=1
            self.assertFalse(helper.try_process_ready_prefill_result(s,b));self.assertEqual(r.copy_done.queries,0)

    def test_unsupported_batch_requests_rejected(self):
        mutations=[lambda s,a,b:setattr(a.forward_mode,'name','DECODE'),
                   lambda s,a,b:setattr(b.forward_mode,'name','MIXED'),
                   lambda s,a,b:setattr(a,'spec_info',object()),
                   lambda s,a,b:setattr(b,'chunked_req',object()),
                   lambda s,a,b:setattr(s,'chunked_req',object()),
                   lambda s,a,b:setattr(a.reqs[0],'pending_bootstrap',True),
                   lambda s,a,b:setattr(a.reqs[0],'inflight_middle_chunks',1),
                   lambda s,a,b:setattr(a.reqs[0],'grammar',object()),
                   lambda s,a,b:setattr(b.reqs[0],'to_finish',object()),
                   lambda s,a,b:setattr(a.reqs[0].extend_range,'end',1)]
        for mutation in mutations:
            s,a,b,r=self.fixture();mutation(s,a,b)
            self.assertFalse(helper.try_process_ready_prefill_result(s,b));self.assertEqual(r.copy_done.queries,0)
        s,a,b,r=self.fixture();self.assertFalse(helper.try_process_ready_prefill_result(s,None))

    def test_logprob_either_batch_rejected_before_query(self):
        for which in ('previous', 'current'):
            s,a,b,r=self.fixture()
            (a if which=='previous' else b).return_logprob=True
            self.assertFalse(helper.try_process_ready_prefill_result(s,b))
            self.assertEqual(r.copy_done.queries,0)
            self.assertIs(s.result_queue[0][0],a)
            self.assertEqual(s.log,[])

    @mock.patch.object(helper, "_supports_cache", return_value=True)
    def test_outer_static_admission(self, cache_predicate):
        # Only outer-control-flow fakes; real cache qualification/lifecycle is
        # covered by the isolated actual CPU-object fixture, not this mock.
        graph=NS(prefill=NS(backend='disabled'));s=scheduler()
        self.assertTrue(helper.supports_early_prefill_result(s,graph,-1))
        self.assertFalse(helper.supports_early_prefill_result(s,None,-1))
        self.assertFalse(helper.supports_early_prefill_result(s,graph,16384))
        for name in vars(s.ps):
            setattr(s.ps,name,2);self.assertFalse(helper.supports_early_prefill_result(s,graph,-1));setattr(s.ps,name,1)
        for name in list(vars(s)):
            if name.startswith('enable_') and name!='enable_overlap':
                setattr(s,name,True);self.assertFalse(helper.supports_early_prefill_result(s,graph,-1));setattr(s,name,False)
        graph.prefill.backend='tc_piecewise'
        self.assertFalse(helper.supports_early_prefill_result(s,graph,-1))

    def test_queue_and_runtime_errors_propagate(self):
        s,a,b,r=self.fixture();s.result_queue.clear()
        with self.assertRaises(IndexError):helper.try_process_ready_prefill_result(s,b)
        s,a,b,r=self.fixture();r.copy_done.error=True
        with self.assertRaisesRegex(RuntimeError,'query failure'):helper.try_process_ready_prefill_result(s,b)
        self.assertIs(s.result_queue[0][0],a)

    def run_loop(self, flag, ready, npu=True):
        s=scheduler();a,b=batch('A',1),batch('B',2);ra,rb=result(Event(ready)),result()
        plans=iter([a,b,None]);calls=0
        s.running_batch=None;s._engine_paused=False;s.waiting_queue=[]
        s.request_receiver=NS(recv_requests=lambda:[])
        s.process_input_requests=lambda r:None
        s.disagg_prefill_bootstrap_queue=NS(pop_bootstrapped=lambda:[])
        s.get_next_disagg_prefill_batch_to_run=lambda **kw:NS(running_batch=None,batch_to_run=next(plans))
        s.ngram_embedding_manager=NS(prepare_for_forward=lambda batch,**kw:batch)
        def run(batch):s.log.append(('run',batch.name));return ra if batch is a else rb
        s.run_batch=run;s._apply_war_barrier=lambda:s.log.append(('war',None))
        s.launch_batch_sample_if_needed=lambda r,batch:s.log.append(('sample',batch.name if batch else None))
        class Done(Exception):pass
        def poll():
            nonlocal calls
            calls+=1;s.log.append(('poll',calls))
            if calls==3:raise Done()
        s.process_disagg_prefill_inflight_queue=poll;s.on_idle=lambda:s.log.append(('idle',None))
        namespace={'deque':deque,'_is_npu':npu,'envs':NS(SGLANG_NPU_PD_PREFILL_EARLY_RESULT=NS(get=lambda:flag)),
                   'get_exec':lambda:NS(graph=NS(cuda_graph_config=NS(prefill=NS(backend='disabled')))),
                   'get_schedule':lambda:NS(chunked_prefill_size=-1),
                   'supports_early_prefill_result':helper.supports_early_prefill_result,
                   'try_process_ready_prefill_result':helper.try_process_ready_prefill_result}
        loop=method(SRT/'disaggregation/prefill.py','SchedulerDisaggregationPrefillMixin',
                    'event_loop_overlap_disagg_prefill',namespace)
        with self.assertRaises(Done):loop(s)
        self.assertEqual([x for x in s.log if x[0]=='process'],[('process','A'),('process','B')])
        self.assertEqual(len(s.result_queue),0)
        return s.log,ra

    @mock.patch.object(helper, "_supports_cache", return_value=True)
    def test_actual_loop_order_and_default_fallback(self, cache_predicate):
        # Cache qualification is separately exercised on real CPU cache objects.
        for flag,ready,npu,early in [(True,True,True,True),(True,False,True,False),
                                     (False,True,True,False),(True,True,False,False)]:
            log,r=self.run_loop(flag,ready,npu)
            self.assertEqual(log.index(('process','A'))<log.index(('run','B')),early)
            if not flag or not npu:self.assertEqual(r.copy_done.queries,0)

    def test_old_radix_and_missing_cache_rejected(self):
        self.assertFalse(helper._supports_cache(scheduler().tree_cache))
        self.assertFalse(helper._supports_cache(None))

    def test_actual_copy_method_marker_only_after_successful_record(self):
        for record_fails in (False,True):
            log=[];owner=NS(copy_done_recorded=True,next_token_ids=object(),
                logits_output=NS(hidden_states=None),accept_lens=None,block_accept_lens=None,
                cap_lens=None,routed_experts_output=None,indexer_topk_output=None,
                expert_distribution_metrics=None,copy_auxiliary_output_to_cpu=lambda:log.append('aux'))
            def record():
                self.assertFalse(owner.copy_done_recorded);log.append('record')
                if record_fails:raise RuntimeError('record failure')
            owner.copy_done=NS(record=record)
            fn=method(SRT/'managers/utils.py','GenerationBatchResult','copy_to_cpu',
                      {'_async_d2h':lambda t:log.append('d2h') or t})
            if record_fails:
                with self.assertRaisesRegex(RuntimeError,'record failure'):fn(owner,False,False)
                self.assertFalse(owner.copy_done_recorded)
            else:fn(owner,False,False);self.assertTrue(owner.copy_done_recorded)
            self.assertEqual(log,['d2h','aux','record'])


if __name__=='__main__':unittest.main()
