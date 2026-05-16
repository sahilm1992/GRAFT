import os
import json
import pandas as pd
from pathlib import Path

def collect_metrics():
    datasets = ['bail', 'income', 'pokec']
    models = ['GCN_MLP', 'GIN_MLP', 'SAGE_MLP', 'GAT_MLP']
    model_map = {
        'GCN_MLP': 'gcn',
        'GIN_MLP': 'gin',
        'SAGE_MLP': 'sage',
        'GAT_MLP': 'gat'
    }
    
    ablated_feature_map = {
        'bail': 'WHITE',
        'income': 'fnlwgt',
        'pokec': 'AGE'
    }
    
    pretrain_root = Path('data/seed_gnn_data/results/seed_gnn')
    edit_root = Path('data/editing_pipelines/leastsquares')
    
    rows = []
    
    for ds in datasets:
        for m_name in models:
            m_lower = model_map[m_name]
            ablated_feat = ablated_feature_map[ds]
            
            # 1. Full Features (Pretrain)
            full_pretrain_path = pretrain_root / m_lower / ds / 'full_features' / 'metrics_pretrain.json'
            if full_pretrain_path.exists():
                with open(full_pretrain_path, 'r') as f:
                    data = json.load(f)
                    m = data['training']['metrics']
                    rows.append({
                        'Dataset': ds,
                        'Model': m_name,
                        'RunType': 'Full Features',
                        'Test_Acc': m.get('final_test_accuracy'),
                        'Test_AUC_PR': m.get('final_test_auc_pr')
                    })
            
            # 2. Feature Ablated (Pretrain)
            ablated_dir = f'no_{ablated_feat}'
            ablated_pretrain_path = pretrain_root / m_lower / ds / ablated_dir / 'metrics_pretrain.json'
            if ablated_pretrain_path.exists():
                with open(ablated_pretrain_path, 'r') as f:
                    data = json.load(f)
                    m = data['training']['metrics']
                    rows.append({
                        'Dataset': ds,
                        'Model': m_name,
                        'RunType': f'No {ablated_feat}',
                        'Test_Acc': m.get('final_test_accuracy'),
                        'Test_AUC_PR': m.get('final_test_auc_pr')
                    })
            
            # 3. Edit Runs
            for edit_suffix in ['mean_sensitivity', 'onlycorrect']:
                edit_path = edit_root / ds / m_name / f'metrics_edit_{edit_suffix}.json'
                if edit_path.exists():
                    with open(edit_path, 'r') as f:
                        data = json.load(f)
                        
                        # Performance
                        m_before = data['metrics_before']['test']
                        m_after = data['metrics_after']['test']
                        
                        # Sensitivity
                        s_before = data['sensitivity_metrics']['before']['test']
                        s_after = data['sensitivity_metrics']['after']['test']
                        
                        # Fairness
                        f_before = data['fairness_metrics']['before']
                        f_after = data['fairness_metrics']['after']
                        
                        common = {
                            'Dataset': ds,
                            'Model': m_name,
                            'RunType': f'Edit {edit_suffix.replace("_", " ").title()}'
                        }
                        
                        # Before Row
                        row_b = common.copy()
                        row_b['RunType'] += ' (Before)'
                        row_b.update({
                            'Test_Acc': m_before.get('acc'),
                            'Test_AUC_PR': m_before.get('auc_pr'),
                            'Sens_MeanVar': s_before.get('mean_var'),
                            'Sens_MeanRelVar': s_before.get('mean_rel_var'),
                            'Sens_MeanFlip': s_before.get('mean_flip_fraction'),
                            'Fair_SP': f_before.get('sp'),
                            'Fair_EO': f_before.get('eo'),
                            'Fair_CF': f_before.get('counterfactual'),
                            'Fair_Instability': f_before.get('instability')
                        })
                        rows.append(row_b)
                        
                        # After Row
                        row_a = common.copy()
                        row_a['RunType'] += ' (After)'
                        row_a.update({
                            'Test_Acc': m_after.get('acc'),
                            'Test_AUC_PR': m_after.get('auc_pr'),
                            'Sens_MeanVar': s_after.get('mean_var'),
                            'Sens_MeanRelVar': s_after.get('mean_rel_var'),
                            'Sens_MeanFlip': s_after.get('mean_flip_fraction'),
                            'Fair_SP': f_after.get('sp'),
                            'Fair_EO': f_after.get('eo'),
                            'Fair_CF': f_after.get('counterfactual'),
                            'Fair_Instability': f_after.get('instability')
                        })
                        rows.append(row_a)

    df = pd.DataFrame(rows)
    output_file = 'all_collected_metrics.csv'
    df.to_csv(output_file, index=False)
    print(f"Collected {len(df)} rows. Saved to {output_file}")

if __name__ == '__main__':
    collect_metrics()



