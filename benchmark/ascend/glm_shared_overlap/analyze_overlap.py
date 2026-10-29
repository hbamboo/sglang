import collections
import csv
import json
import pathlib
import statistics
import argparse
from decimal import Decimal

parser=argparse.ArgumentParser()
parser.add_argument('--customer', action='store_true')
parser.add_argument('--optimized-prefix')
args=parser.parse_args()
suffix='-customer' if args.customer else ''

ROOT = pathlib.Path(__file__).parent
RESULTS = ROOT / 'results'
profiles = {}
for variant in ['baseline', 'optimized']:
    prefix=args.optimized_prefix if variant=='optimized' and args.optimized_prefix else variant+suffix
    path = next((RESULTS / (prefix + '-profile')).rglob('kernel_details.csv'))
    rows = list(csv.DictReader(path.open(newline='', encoding='utf-8-sig')))
    origin=min(Decimal(row['Start Time(us)'].strip()) for row in rows)
    for row in rows:
        row['start_us'] = float(Decimal(row['Start Time(us)'].strip())-origin)
        row['duration_us'] = float(row['Duration(us)'])
        row['end_us'] = row['start_us'] + row['duration_us']
    rows.sort(key=lambda row: row['start_us'])
    profiles[variant] = rows

def selected(rows, predicate):
    return sorted([row for row in rows if predicate(row)], key=lambda row:row['start_us'])

def overlap_us(interval, targets):
    return sum(max(0, min(interval['end_us'], target['end_us']) - max(interval['start_us'], target['start_us'])) for target in targets)

opt = profiles['optimized']
main = collections.Counter(row['Stream ID'] for row in opt).most_common(1)[0][0]
gateups = selected(opt, lambda r:r['Type']=='QuantBatchMatmulV3' and ';3072,2048;' in r['Input Shapes'] and r['Stream ID']!=main)
assert len(gateups)>0 and len(gateups)%46==0
token_counts={int(r['Input Shapes'].strip('"').split(',')[0]) for r in gateups}
def has_rows(row, counts):
    return int(row['Input Shapes'].strip('"').split(',')[0]) in counts
aux = gateups[0]['Stream ID']
assert main != aux and all(row['Stream ID']==aux for row in gateups)
topks = selected(opt, lambda r:r['Type']=='MoeGatingTopK' and has_rows(r,token_counts))
inits = selected(opt, lambda r:r['Type']=='MoeInitRoutingV3' and has_rows(r,token_counts))
gmm1s = selected(opt, lambda r:r['Type']=='GroupedMatmulSwigluQuantV2' and has_rows(r,{n*4 for n in token_counts}))
gmm2s = selected(opt, lambda r:r['Type']=='GroupedMatmul' and has_rows(r,{n*4 for n in token_counts}))
finals = selected(opt, lambda r:r['Type']=='MoeFinalizeRoutingV2' and has_rows(r,{n*4 for n in token_counts}))
downs = selected(opt, lambda r:r['Type']=='QuantBatchMatmulV3' and r['Stream ID']==aux and ';2048,1536;' in r['Input Shapes'])
swiglus = selected(opt, lambda r:r['Type']=='SwiGlu' and r['Stream ID']==aux)
quants = selected(opt, lambda r:r['Name']=='DynamicMxQuant' and r['Stream ID']==aux and ',1536"' in r['Input Shapes'])
assert all(len(items)==len(gateups) for items in [topks,inits,gmm1s,gmm2s,finals,downs,swiglus,quants])
records=[]
for index in range(len(gateups)):
    gate, topk, init, gmm1, gmm2, final, down, swiglu, quant = [items[index] for items in [gateups,topks,inits,gmm1s,gmm2s,finals,downs,swiglus,quants]]
    activation=dict(start_us=swiglu['start_us'],end_us=quant['end_us'])
    gate_overlap=overlap_us(gate,[topk,init])
    activation_overlap=overlap_us(activation,[gmm2])
    down_overlap=overlap_us(down,[final])
    assert swiglu['start_us']>=gmm1['end_us']-0.5
    assert down['start_us']>=gmm2['end_us']-0.5
    records.append(dict(prefill=index//46,layer=index%46+1,tokens=int(gate['Input Shapes'].strip('"').split(',')[0]),gate_us=gate['duration_us'],gate_overlap_us=gate_overlap,gate_hidden_pct=min(100,gate_overlap/gate['duration_us']*100),act_quant_span_us=activation['end_us']-activation['start_us'],act_quant_overlap_us=activation_overlap,down_us=down['duration_us'],down_overlap_us=down_overlap,down_hidden_pct=min(100,down_overlap/down['duration_us']*100),topk_us=topk['duration_us'],init_us=init['duration_us'],gmm2_us=gmm2['duration_us'],finalize_us=final['duration_us']))
with (RESULTS/('overlap'+suffix+'-by-layer.csv')).open('w',newline='',encoding='utf-8-sig') as file:
    writer=csv.DictWriter(file,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
summary=dict(main_stream=main,shared_stream=aux,layers=len(records),all_three_stages_overlap=all(r['gate_overlap_us']>0 and r['act_quant_overlap_us']>0 and r['down_overlap_us']>0 for r in records),gate_overlap_ms=sum(r['gate_overlap_us'] for r in records)/1000,act_quant_overlap_ms=sum(r['act_quant_overlap_us'] for r in records)/1000,down_overlap_ms=sum(r['down_overlap_us'] for r in records)/1000,gate_hidden_mean_pct=statistics.mean(r['gate_hidden_pct'] for r in records),down_hidden_mean_pct=statistics.mean(r['down_hidden_pct'] for r in records),prefill_batches=[],variants={})
for prefill in sorted({r['prefill'] for r in records}):
    group=[r for r in records if r['prefill']==prefill]
    summary['prefill_batches'].append(dict(index=prefill,tokens=group[0]['tokens'],layers=len(group),positive_overlap_layer_counts=[sum(r[key]>0 for r in group) for key in ['gate_overlap_us','act_quant_overlap_us','down_overlap_us']],overlap_ms=[sum(r[key] for r in group)/1000 for key in ['gate_overlap_us','act_quant_overlap_us','down_overlap_us']]))
for variant, rows in profiles.items():
    times=collections.defaultdict(float)
    for row in rows:times[row['Type']]+=row['duration_us']/1000
    summary['variants'][variant]=dict(kernel_sum_ms=sum(r['duration_us'] for r in rows)/1000,device_span_ms=(max(r['end_us'] for r in rows)-min(r['start_us'] for r in rows))/1000,by_type_ms=dict(times))
if not args.customer:
    base=json.loads((RESULTS/'baseline-ttft.json').read_text(encoding='utf-8'))
    candidate=json.loads((RESULTS/'optimized-ttft.json').read_text(encoding='utf-8'))
    assert base['input_sha256']==candidate['input_sha256']
    summary['ttft']=dict(baseline=base['summary'],optimized=candidate['summary'],median_reduction_pct=(1-candidate['summary']['median_ms']/base['summary']['median_ms'])*100,mean_reduction_pct=(1-candidate['summary']['mean_ms']/base['summary']['mean_ms'])*100,all_first_tokens_identical=len({tuple(r['result']['output_ids']) for data in [base,candidate] for r in data['records']})==1)
(RESULTS/('overlap'+suffix+'-analysis.json')).write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='variants'},indent=2))
for name in ['MoeGatingTopK','MoeInitRoutingV3','GroupedMatmul','MoeFinalizeRoutingV2']:
    print(name,summary['variants']['baseline']['by_type_ms'][name],summary['variants']['optimized']['by_type_ms'][name])
