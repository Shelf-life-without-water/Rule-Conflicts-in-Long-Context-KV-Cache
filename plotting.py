
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_probe_accuracy(df: pd.DataFrame, title: str='Layerwise probe accuracy'):
    plt.figure(figsize=(9, 4))
    for rep_field, sub in df.groupby('rep_field'):
        sub_all = sub[sub['slot'] == '__all__'].sort_values('layer')
        if len(sub_all) == 0:
            continue
        plt.plot(sub_all['layer'], sub_all['test_acc'], marker='o', label=rep_field)
    plt.xlabel('Layer')
    plt.ylabel('Test accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def summarize_erase_results(df: pd.DataFrame):
    metrics = ['ifc', 'cr', 'cfr', 'payload_ok', 'current_slot_acc', 'stable_slot_acc', 'ppl_gold']
    agg = df.groupby(['method', 'target', 'rank', 'alpha', 'slot_specific'])[metrics].mean().reset_index()
    return agg.sort_values(['method', 'target', 'rank', 'alpha', 'slot_specific'])
