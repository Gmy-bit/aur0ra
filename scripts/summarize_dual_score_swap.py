#!/usr/bin/env python3
"""Summarize native/swapped scores of HyCoCLIP and Aurora (HyCoSLIP)."""
import json
import sys
from pathlib import Path

def main():
    output_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('eval_logs/dual_score_swap')
    model_names = [sys.argv[2]] if len(sys.argv) > 2 else ['hycoclip', 'aurora']
    for model in model_names:
        path = output_root/model/'results.json'
        if not path.is_file():
            print(f'[{model}] No results yet: {path}')
            continue
        records = json.loads(path.read_text(encoding='utf8'))
        if not isinstance(records, list): records=[records]
        scores = {'lorentz_distance':{}, 'root_angle':{}}
        for rec in records:
            if rec.get('model',model)!=model: continue
            mode=rec.get('eval_score_type')
            if mode in scores:
                for metric, val in rec.get('results',{}).items():
                    if metric in ('imagenet','coco_t2i_r5'):
                        scores[mode][metric]=float(val)
        native = 'lorentz_distance' if model=='hycoclip' else 'root_angle'
        swapped = 'root_angle' if model=='hycoclip' else 'lorentz_distance'
        print(f'\n=== {model}: {native} (native) -> {swapped} (swapped) ===')
        print(f'{"Metric":<24}{"Native":>16}{"Swapped":>16}{"Delta":>14}')
        for met in ('imagenet','coco_t2i_r5'):
            before = scores[native].get(met)
            after  = scores[swapped].get(met)
            def fmt(x): return f'{x:.3f}' if x is not None else '[not run]'
            delta = f'{after-before:+.3f}' if before is not None and after is not None else '--'
            print(f'{met:<24}{fmt(before):>16}{fmt(after):>16}{delta:>14}')
        print(f'Data: {path}')

if __name__=='__main__': main()
