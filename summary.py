# compare_fixed.py
import json
import pandas as pd
from pathlib import Path

def extract_metrics(data):
    """Robust metric extraction - handles any structure"""
    metrics = {}
    
    # Uncertainty
    unc = data.get('uncertainty', {})
    for key in ['ece', 'brier_score', 'ace', 'inference_time']:
        if key in unc:
            val = unc[key]
            mean = float(val.get('mean', 0))
            std = float(val.get('var', 0)**0.5)
            metrics[f"{key}_mean"] = f"{mean:.4f}"
            metrics[f"{key}_std"] = f"±{std:.4f}"
    
    # Link Prediction
    lp = data.get('link_prediction', {})
    for key in ['mrr', 'hits@1', 'hits@3', 'hits@10', 'inference_time']:
        if key in lp:
            val = lp[key]
            mean = float(val.get('mean', 0))
            std = float(val.get('var', 0)**0.5)
            metrics[f"LP_{key}_mean"] = f"{mean:.4f}"
            metrics[f"LP_{key}_std"] = f"±{std:.4f}"
    
    return metrics

def compare_model(model_name, dataset="WN18RR", results_root="results_lp_uncertainty"):
    model_path = Path(results_root) / dataset / model_name
    all_results = []
    
    print(f"📂 Scanning: {model_path}")
    
    for metrics_file in model_path.rglob("**/metrics.json"):
        print(f"📄 {metrics_file}")
        try:
            with open(metrics_file) as f:
                data = json.load(f)
            
            method = data.get("calibration_method", "unknown")
            ctype = data.get("calibration_type", "baseline")
            label = f"{method}/{ctype}" if method != "none" else "baseline"
            
            metrics = extract_metrics(data)
            metrics['Method'] = label
            metrics['File'] = str(metrics_file)
            all_results.append(metrics)
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    if all_results:
        df = pd.DataFrame(all_results)
        print("\n" + "="*120)
        print(f"📊 COMPLETE RESULTS: {model_name}")
        print("="*120)
        print(df[['Method', 'ece_mean', 'ece_std', 'brier_score_mean', 'brier_score_std', 
                 'ace_mean', 'ace_std', 'inference_time_mean', 'inference_time_std']].round(4))
        
        df.to_csv(f"{model_name.replace('/', '_').replace('.', '_')}_full.csv", index=False)
        print(f"\n💾 Full CSV saved!")
    else:
        print("❌ No valid JSON files found!")

if __name__ == "__main__":
    model = "WN18RR_checkpoint_label_0.9_negative_sampling_3_edge_dropout_0.2_epoch_1700.pth"
    compare_model(model)
