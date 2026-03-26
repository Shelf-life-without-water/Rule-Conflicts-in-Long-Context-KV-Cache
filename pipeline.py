from __future__ import annotations

import argparse
from typing import Any, Dict

import pandas as pd
from tqdm.auto import tqdm

from .analysis import collect_temporal_reps
from .benchmark import build_interp_splits
from .config import CFG
from .data_utils import build_default_distractor_pool
from .env import ARTIFACT_ROOT, RESULTS_ROOT, RUNS_ROOT, SEED
from .erase import run_causal_erase_suite
from .export_utils import export_experiment_summary_json, make_results_archive, torch_load_any
from .metrics import evaluate_none
from .model_utils import cleanup_bundle, load_model_bundle
from .plotting import plot_probe_accuracy, summarize_erase_results
from .probes import run_layerwise_probe_suite
from .records import collect_representation_records
from .span_utils import prepare_segmented_sample
from .subspace import fit_conflict_subspace_from_records, pairwise_subspace_similarity

def build_runtime():
    distractor_pool = build_default_distractor_pool()
    interp_splits = build_interp_splits(CFG, distractor_pool=distractor_pool, seed=SEED)
    return {"distractor_pool": distractor_pool, "interp_splits": interp_splits}

def run_quick_sanity(bundle, interp_splits):
    sample = interp_splits["probe_train"][0]
    prepared = prepare_segmented_sample(bundle, sample, max_context_tokens=bundle.spec.max_context_tokens)
    baseline = evaluate_none(bundle, sample)
    return {"sample": sample, "prepared": prepared, "baseline": baseline}

def collect_or_load_probe_records(bundle, interp_splits):
    probe_layers = list(range(max(0, bundle.num_layers - 8), bundle.num_layers))
    probe_train_records_path = ARTIFACT_ROOT / "probe_train_records.pt"
    probe_test_records_path = ARTIFACT_ROOT / "probe_test_records.pt"
    if probe_train_records_path.exists():
        probe_train_records = torch_load_any(probe_train_records_path)
    else:
        probe_train_records = collect_representation_records(bundle=bundle, samples=interp_splits["probe_train"], layers=probe_layers, split_name="probe_train", save_path=probe_train_records_path)
    if probe_test_records_path.exists():
        probe_test_records = torch_load_any(probe_test_records_path)
    else:
        probe_test_records = collect_representation_records(bundle=bundle, samples=interp_splits["probe_test"], layers=probe_layers, split_name="probe_test", save_path=probe_test_records_path)
    return {"probe_layers": probe_layers, "probe_train_records": probe_train_records, "probe_test_records": probe_test_records}

def run_existence_analysis(probe_train_records, probe_test_records, plot: bool = False):
    probe_df_v = run_layerwise_probe_suite(probe_train_records, probe_test_records, rep_field="V_edit", out_csv=RUNS_ROOT / "probe_suite_V.csv")
    probe_df_k = run_layerwise_probe_suite(probe_train_records, probe_test_records, rep_field="K_edit", out_csv=RUNS_ROOT / "probe_suite_K.csv")
    probe_df_kv = run_layerwise_probe_suite(probe_train_records, probe_test_records, rep_field="KV_edit", out_csv=RUNS_ROOT / "probe_suite_KV.csv")
    probe_df_all = pd.concat([probe_df_v, probe_df_k, probe_df_kv], axis=0, ignore_index=True)
    probe_df_all.to_csv(RUNS_ROOT / "probe_suite_all.csv", index=False)
    if plot:
        plot_probe_accuracy(probe_df_all, title="Conflict probe accuracy by layer")
    return probe_df_all

def run_slot_similarity_analysis(probe_train_records):
    probe_layers = sorted(set(r["layer"] for r in probe_train_records))
    rows = []
    for layer in probe_layers:
        u_global = fit_conflict_subspace_from_records(probe_train_records, layer=layer, rep_field="V_edit", slot=None, rank=4)
        if u_global is None:
            continue
        for slot in CFG.active_slots:
            u_slot = fit_conflict_subspace_from_records(probe_train_records, layer=layer, rep_field="V_edit", slot=slot, rank=4)
            if u_slot is None:
                continue
            rows.append({"layer": layer, "slot": slot, "sim_to_global": pairwise_subspace_similarity(u_global, u_slot)})
    slot_sim_df = pd.DataFrame(rows)
    slot_sim_df.to_csv(RUNS_ROOT / "slot_global_similarity.csv", index=False)
    return slot_sim_df

def run_temporal_analysis(bundle, interp_splits):
    probe_layers = list(range(max(0, bundle.num_layers - 8), bundle.num_layers))
    temporal_rows = []
    for sample in tqdm(interp_splits["probe_train"], desc="temporal reps [probe_train full]"):
        temporal_rows.append(collect_temporal_reps(bundle, sample, layers=probe_layers))
    temp_df = pd.concat(temporal_rows, axis=0, ignore_index=True)
    temp_df.to_csv(RUNS_ROOT / "temporal_reps_full.csv", index=False)
    return temp_df

def run_full_erase_analysis(bundle, probe_train_records, interp_splits):
    erase_out_csv = RUNS_ROOT / "causal_erase_results_full.csv"
    erase_df = run_causal_erase_suite(bundle=bundle, probe_records=probe_train_records, erase_samples=interp_splits["erase_test"], layers=list(range(max(0, bundle.num_layers - 8), bundle.num_layers)), targets=("V", "K", "KV"), ranks=CFG.rank_grid, alphas=CFG.alpha_grid, slot_specific_options=(True, False), out_csv=erase_out_csv)
    summary_erase = summarize_erase_results(erase_df)
    summary_erase.to_csv(RUNS_ROOT / "causal_erase_summary_full.csv", index=False)
    return erase_df, summary_erase

def run_full_pipeline(model_name: str = "llama_8b", plot: bool = False) -> Dict[str, Any]:
    runtime = build_runtime()
    bundle = load_model_bundle(model_name)
    try:
        sanity = run_quick_sanity(bundle, runtime["interp_splits"])
        probe_state = collect_or_load_probe_records(bundle, runtime["interp_splits"])
        probe_df_all = run_existence_analysis(probe_state["probe_train_records"], probe_state["probe_test_records"], plot=plot)
        slot_sim_df = run_slot_similarity_analysis(probe_state["probe_train_records"])
        temp_df = run_temporal_analysis(bundle, runtime["interp_splits"])
        erase_df, summary_erase = run_full_erase_analysis(bundle, probe_state["probe_train_records"], runtime["interp_splits"])
        summary_obj, summary_json_path = export_experiment_summary_json(results_root=RESULTS_ROOT, artifact_root=ARTIFACT_ROOT, runs_root=RUNS_ROOT, cfg=CFG, active_models=[model_name], interp_splits=runtime["interp_splits"], out_path=RESULTS_ROOT / "experiment_summary.json")
        archive_path = make_results_archive(RESULTS_ROOT)
        return {"runtime": runtime, "sanity": sanity, "probe_state": probe_state, "probe_df_all": probe_df_all, "slot_sim_df": slot_sim_df, "temp_df": temp_df, "erase_df": erase_df, "summary_erase": summary_erase, "summary_obj": summary_obj, "summary_json_path": summary_json_path, "archive_path": archive_path}
    finally:
        cleanup_bundle(bundle)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama_8b")
    parser.add_argument("--stage", choices=["full", "sanity"], default="full")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.stage == "full":
        result = run_full_pipeline(model_name=args.model, plot=args.plot)
        print(result["summary_json_path"])
        print(result["archive_path"])
        return
    runtime = build_runtime()
    bundle = load_model_bundle(args.model)
    try:
        result = run_quick_sanity(bundle, runtime["interp_splits"])
        print(result["sample"]["sample_id"])
        print(result["baseline"])
    finally:
        cleanup_bundle(bundle)

if __name__ == "__main__":
    main()
