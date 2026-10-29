"""Rebuild actual MLP windows and normalized physical overlap from CSV."""
import argparse
import bisect
import collections
import csv
from decimal import Decimal
import json
import pathlib
import statistics

parser=argparse.ArgumentParser()
parser.add_argument('--results-dir',type=pathlib.Path,default=pathlib.Path(__file__).parent/'results')
parser.add_argument('--baseline',default='baseline-expandable-customer')
parser.add_argument('--optimized',default='optimized-expandable-customer')
parser.add_argument('--discard-first-requests',type=int,default=2)
parser.add_argument('--request-tokens',type=int,default=32467)
args=parser.parse_args()

def shape_rows(row):
    return int(row['Input Shapes'].strip('"').split(',')[0])

def window(start,end):return dict(start=start,end=end)

def intersection(a,b):return max(0,min(a['end'],b['end'])-max(a['start'],b['start']))

def normalized(a,b):
    return intersection(a,b)/min(a['end']-a['start'],b['end']-b['start'])

result={}
for variant,prefix in [('baseline',args.baseline),('optimized',args.optimized)]:
    path=next((args.results_dir/(prefix+'-profile')).rglob('kernel_details.csv'))
    rows=list(csv.DictReader(path.open(newline='',encoding='utf-8-sig')))
    origin=min(Decimal(r['Start Time(us)'].strip()) for r in rows)
    for r in rows:
        r['start']=float(Decimal(r['Start Time(us)'].strip())-origin)
        r['duration']=float(r['Duration(us)']);r['end']=r['start']+r['duration']
    rows.sort(key=lambda r:r['start'])
    main=collections.Counter(r['Stream ID'] for r in rows).most_common(1)[0][0]
    topk=[r for r in rows if r['Type']=='MoeGatingTopK' and shape_rows(r)>128]
    counts={shape_rows(r) for r in topk}
    gate=[r for r in rows if r['Type']=='QuantBatchMatmulV3' and ';3072,2048;' in r['Input Shapes'] and shape_rows(r) in counts]
    down=[r for r in rows if r['Type']=='QuantBatchMatmulV3' and ';2048,1536;' in r['Input Shapes'] and shape_rows(r) in counts]
    act=[r for r in rows if r['Type']=='SwiGlu' and ',3072"' in r['Input Shapes'] and shape_rows(r) in counts]
    init=[r for r in rows if r['Type']=='MoeInitRoutingV3' and shape_rows(r) in counts]
    gmm2=[r for r in rows if r['Type']=='GroupedMatmul' and shape_rows(r) in {n*4 for n in counts}]
    final=[r for r in rows if r['Type']=='MoeFinalizeRoutingV2' and shape_rows(r) in {n*4 for n in counts}]
    norms=[r for r in rows if r['Type']=='AddRmsNorm' and r['Stream ID']==main]
    additions=[r for r in rows if r['Name']=='aclnnAdd_AddAiCore_Add' and r['Stream ID']==main]
    quant=[r for r in rows if r['Name']=='DynamicMxQuant' and any(r['Input Shapes']=='"'+str(n)+',2048"' for n in counts)]
    quant_by_stream={s:[r for r in quant if r['Stream ID']==s] for s in {r['Stream ID'] for r in quant}}
    norm_times=[r['start'] for r in norms];add_times=[r['start'] for r in additions]
    quant_times={s:[r['start'] for r in qq] for s,qq in quant_by_stream.items()}
    assert len(topk)>0 and len(topk)%46==0
    assert all(len(seq)==len(topk) for seq in [gate,down,act,init,gmm2,final]),[len(seq) for seq in [topk,gate,down,act,init,gmm2,final]]
    records=[];request=0;request_tokens=0
    for index in range(len(topk)):
        tokens=shape_rows(topk[index])
        if index%46==0:
            batch_request=request
            request_tokens+=tokens
            assert request_tokens<=args.request_tokens,(request_tokens,args.request_tokens)
            if request_tokens==args.request_tokens:request+=1;request_tokens=0
        norm=norms[bisect.bisect_left(norm_times,topk[index]['start'])-1]
        addition=additions[bisect.bisect_left(add_times,final[index]['end'])]
        s=gate[index]['Stream ID'];q=quant_by_stream[s][bisect.bisect_left(quant_times[s],gate[index]['start'])-1]
        side1=window(q['start'],gate[index]['end']);main1=window(topk[index]['start'],init[index]['end'])
        side2=window(act[index]['start'],down[index]['start']);main2=window(gmm2[index]['start'],gmm2[index]['end'])
        # The second stage includes only activation/quant work, excluding the
        # intentional wait for GMM2 before down. Take the last quant end.
        q2=[r for r in rows if r['Name']=='DynamicMxQuant' and r['Stream ID']==s and act[index]['end']<=r['start']<=down[index]['start'] and r['Input Shapes']=='"'+str(tokens)+',1536"']
        assert len(q2)==1
        side2['end']=q2[0]['end']
        side3=window(down[index]['start'],down[index]['end']);main3=window(final[index]['start'],final[index]['end'])
        pct=[normalized(a,b) if s!=main else 0 for a,b in [(side1,main1),(side2,main2),(side3,main3)]]
        records.append(dict(request=batch_request,prefill=index//46,layer=index%46+1,tokens=tokens,selected=batch_request>=args.discard_first_requests,module_us=addition['end']-norm['end'],gate_overlap_pct=pct[0],act_quant_overlap_pct=pct[1],down_overlap_pct=pct[2],init_us=init[index]['duration'],gmm2_us=gmm2[index]['duration'],finalize_us=final[index]['duration']))
    assert request_tokens==0
    with (args.results_dir/(prefix+'-mlp-windows.csv')).open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    groups=[]
    for tokens in sorted(counts):
        group=[r for r in records if r['selected'] and r['tokens']==tokens]
        if not group:continue
        groups.append(dict(tokens=tokens,instances=len(group),module_median_us=statistics.median(r['module_us'] for r in group),module_mean_us=statistics.mean(r['module_us'] for r in group),true_overlap_counts=[sum(r[k]>=0.5 for r in group) for k in ['gate_overlap_pct','act_quant_overlap_pct','down_overlap_pct']],median_overlap_pct=[statistics.median(r[k] for r in group) for k in ['gate_overlap_pct','act_quant_overlap_pct','down_overlap_pct']],init_mean_us=statistics.mean(r['init_us'] for r in group),gmm2_mean_us=statistics.mean(r['gmm2_us'] for r in group),finalize_mean_us=statistics.mean(r['finalize_us'] for r in group)))
    result[variant]=dict(prefix=prefix,request_passes=request,discarded=args.discard_first_requests,groups=groups)
(args.results_dir/(args.optimized+'-mlp-analysis.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
