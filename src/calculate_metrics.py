#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
상세 메트릭 계산: Precision, Recall, F1-Score
"""

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def calculate_detailed_metrics(csv_path, method_name):
    """상세 메트릭 계산"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    
    print(f"\n{'='*70}")
    print(f"📊 {method_name} - Detailed Metrics")
    print(f"{'='*70}")
    
    # 기본 정확도
    accuracy = df['acc'].mean()
    print(f"\nOverall Accuracy: {accuracy:.4f} ({df['acc'].sum():.0f}/{len(df)})")
    
    # Classification Report
    print(f"\n{'-'*70}")
    print("Classification Report:")
    print(f"{'-'*70}")
    
    report = classification_report(
        df['gt'], 
        df['pred'], 
        output_dict=True,
        zero_division=0
    )
    
    # 카테고리별 메트릭
    categories = sorted(df['gt'].unique())
    
    print(f"\n{'Category':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print(f"{'-'*70}")
    
    for cat in categories:
        if cat in report:
            metrics = report[cat]
            print(f"{cat:<20} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                  f"{metrics['f1-score']:<12.4f} {int(metrics['support']):<10}")
    
    # Macro/Weighted Average
    print(f"{'-'*70}")
    print(f"{'Macro Avg':<20} {report['macro avg']['precision']:<12.4f} "
          f"{report['macro avg']['recall']:<12.4f} {report['macro avg']['f1-score']:<12.4f}")
    print(f"{'Weighted Avg':<20} {report['weighted avg']['precision']:<12.4f} "
          f"{report['weighted avg']['recall']:<12.4f} {report['weighted avg']['f1-score']:<12.4f}")
    
    # Confusion Matrix
    print(f"\n{'-'*70}")
    print("Confusion Matrix:")
    print(f"{'-'*70}")
    
    cm = confusion_matrix(df['gt'], df['pred'], labels=categories)
    
    # Header
    print(f"\n{'True/Pred':<20}", end="")
    for cat in categories:
        print(f"{cat[:10]:<12}", end="")
    print()
    print(f"{'-'*70}")
    
    # Rows
    for i, true_cat in enumerate(categories):
        print(f"{true_cat:<20}", end="")
        for j, pred_cat in enumerate(categories):
            print(f"{cm[i][j]:<12}", end="")
        print()
    
    return report


def compare_all_methods():
    """모든 방법 비교"""
    
    files = {
        "Baseline (LLM only)": "./runs/results_baseline_176.csv",
        "Agentic (Retrieval + LLM)": "./runs/results_agentic_176.csv",
        "Hybrid (Rule + Retrieval)": "./runs/results_hybrid_high_performance.csv",
    }
    
    print("\n" + "="*70)
    print("📈 COMPREHENSIVE METRICS COMPARISON")
    print("="*70)
    
    all_reports = {}
    for name, path in files.items():
        try:
            report = calculate_detailed_metrics(path, name)
            all_reports[name] = report
        except Exception as e:
            print(f"\n⚠️  Error loading {name}: {e}")
    
    # Summary Table
    print("\n" + "="*70)
    print("📊 SUMMARY COMPARISON")
    print("="*70)
    
    print(f"\n{'Method':<35} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12}")
    print(f"{'-'*70}")
    
    for name, report in all_reports.items():
        if report:
            # Accuracy는 weighted avg recall과 동일
            accuracy = report['weighted avg']['recall']
            macro_f1 = report['macro avg']['f1-score']
            weighted_f1 = report['weighted avg']['f1-score']
            
            print(f"{name:<35} {accuracy:<12.4f} {macro_f1:<12.4f} {weighted_f1:<12.4f}")
    
    print("="*70)


if __name__ == "__main__":
    compare_all_methods()
