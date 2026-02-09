#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
고성능 4개 카테고리만 선택해서 대규모 평가
- TV, Audio, Vehicle/Autos, Advanced Tech
- 각 카테고리 최대한 많이 샘플링
"""

import argparse
import json
import os
import random
from typing import Dict, List
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except Exception:
    faiss = None
    SentenceTransformer = None


# Helper functions
def get_field(d: Dict, *keys):
    for k in keys:
        if k in d:
            return d[k]
    return ""

def get_label(r: Dict, label_field: str) -> str:
    v = None
    for key in r.keys():
        if key.lower() == label_field.lower():
            v = r[key]
            break
    
    if v is None or v == "":
        return "UNKNOWN"
    return str(v)

def read_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except:
                    continue
    return out

def build_doc_text(r: Dict) -> str:
    headline = get_field(r, "headline", "Headline", "Title") or ""
    summary = get_field(r, "summary_en", "Summary_en", "Summary") or ""
    keywords = get_field(r, "keywords", "Keywords") or ""
    products = get_field(r, "products", "Products") or ""
    
    return f"{headline} {summary} {keywords} {products}".lower()


# Rule-based Classifier
class RuleBasedClassifier:
    def __init__(self):
        self.rules = {
            "TV": [
                "oled", "qled", "tv", "television", "display", "screen", "webos",
                "4k", "8k", "smart tv", "gaming tv"
            ],
            "Audio": [
                "speaker", "soundbar", "audio", "sound", "xboom", "tone free",
                "earbuds", "headphone"
            ],
            "Vehicle/Autos": [
                "vehicle", "automotive", "car", "auto", "sdv", "motortrend",
                "vs company", "vehicle solution", "infotainment"
            ],
            "Advanced Tech": [
                "ai", "artificial intelligence", "robot", "thinq", "deep learning",
                "machine learning", "smart home", "iot", "ces 2025"
            ],
        }
    
    def classify(self, text: str) -> Dict:
        text = text.lower()
        
        scores = {}
        for category, keywords in self.rules.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[category] = score
        
        if not scores:
            return {"label": "UNKNOWN", "confidence": 0.0, "method": "rule"}
        
        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]
        total_score = sum(scores.values())
        
        confidence = best_score / total_score if total_score > 0 else 0.0
        
        return {
            "label": best_label,
            "confidence": confidence,
            "method": "rule",
            "scores": scores
        }


# Retrieval-based Classifier
class RetrievalClassifier:
    def __init__(self, index_dir: str, encoder_name: str):
        if faiss is None or SentenceTransformer is None:
            raise RuntimeError("faiss / sentence-transformers not installed.")
        
        self.index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        self.meta = pd.read_parquet(os.path.join(index_dir, "meta.parquet"))
        
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        self.encoder = SentenceTransformer(encoder_name, local_files_only=True)
        
        print(f"✅ Retrieval Classifier 초기화 완료 ({len(self.meta)} docs)")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        v = self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        return v
    
    def classify(self, text: str, top_k: int = 10) -> Dict:
        qv = self.embed([text])
        scores, ids = self.index.search(qv, top_k)
        ids = ids[0].tolist()
        scores = scores[0].tolist()
        
        label_votes = Counter()
        label_scores = defaultdict(float)
        
        for doc_id, sc in zip(ids, scores):
            if doc_id < 0 or doc_id >= len(self.meta):
                continue
            row = self.meta.iloc[doc_id]
            label = row.get('label', 'UNKNOWN')
            if label != 'UNKNOWN':
                label_votes[label] += 1
                label_scores[label] += sc
        
        if not label_votes:
            return {"label": "UNKNOWN", "confidence": 0.0, "method": "retrieval"}
        
        best_label = label_votes.most_common(1)[0][0]
        best_votes = label_votes[best_label]
        total_votes = sum(label_votes.values())
        
        confidence = best_votes / total_votes if total_votes > 0 else 0.0
        
        return {
            "label": best_label,
            "confidence": confidence,
            "method": "retrieval",
            "votes": dict(label_votes)
        }


# Hybrid Classifier
class HybridClassifier:
    def __init__(self, rule_classifier: RuleBasedClassifier, retrieval_classifier: RetrievalClassifier):
        self.rule = rule_classifier
        self.retrieval = retrieval_classifier
    
    def classify(self, text: str, rule_threshold: float = 0.5) -> Dict:
        rule_result = self.rule.classify(text)
        
        if rule_result["confidence"] >= rule_threshold:
            return rule_result
        
        retrieval_result = self.retrieval.classify(text)
        
        if retrieval_result["confidence"] > rule_result["confidence"]:
            return retrieval_result
        
        return rule_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/A_company_202501_clean.jsonl")
    parser.add_argument("--index_dir", default="index")
    parser.add_argument("--output_dir", default="runs")
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 고성능 카테고리만 선택
    high_performance_categories = ['TV', 'Audio', 'Vehicle/Autos', 'Advanced Tech']
    
    # 데이터 로드
    rows = read_jsonl(args.data)
    
    # 고성능 카테고리만 필터링
    category_samples = defaultdict(list)
    for r in rows:
        label = get_label(r, "Article_Category")
        if label in high_performance_categories:
            category_samples[label].append(r)
    
    print(f"\n📊 고성능 카테고리 샘플 수:")
    for cat in high_performance_categories:
        print(f"   {cat:20s}: {len(category_samples[cat])}개")
    
    # 각 카테고리에서 최대한 많이 샘플링 (최대 50개)
    test_samples = []
    for cat in high_performance_categories:
        available = category_samples[cat]
        n_samples = min(50, len(available))
        sampled = random.sample(available, n_samples)
        test_samples.extend(sampled)
        print(f"   → {cat}: {n_samples}개 샘플링")
    
    print(f"\n✅ 총 {len(test_samples)}개 샘플 선택")
    
    # Classifiers 초기화
    rule_clf = RuleBasedClassifier()
    retrieval_clf = RetrievalClassifier(args.index_dir, args.encoder)
    hybrid_clf = HybridClassifier(rule_clf, retrieval_clf)
    
    # 평가
    results = []
    for i, sample in enumerate(tqdm(test_samples, desc="Evaluating")):
        headline = get_field(sample, "headline", "Headline", "Title")
        summary = get_field(sample, "summary_en", "Summary_en", "Summary")
        keywords = get_field(sample, "keywords", "Keywords")
        products = get_field(sample, "products", "Products")
        gt_label = get_label(sample, "Article_Category")
        
        text = f"{headline} {summary} {keywords} {products}"
        
        try:
            pred = hybrid_clf.classify(text, rule_threshold=0.4)
            pred_label = pred.get("label", "UNKNOWN")
            confidence = pred.get("confidence", 0.0)
            method = pred.get("method", "unknown")
        except Exception as e:
            print(f"⚠️  Error: {e}")
            pred_label = "UNKNOWN"
            confidence = 0.0
            method = "error"
        
        acc = 1 if pred_label == gt_label else 0
        
        results.append({
            "idx": i,
            "headline": headline,
            "gt": gt_label,
            "pred": pred_label,
            "acc": acc,
            "confidence": confidence,
            "method": method
        })
    
    # 결과 저장
    df = pd.DataFrame(results)
    os.makedirs(args.output_dir, exist_ok=True)
    output_csv = os.path.join(args.output_dir, "results_hybrid_high_performance.csv")
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    
    # 통계
    accuracy = df['acc'].mean()
    print(f"\n{'='*60}")
    print(f"📊 고성능 카테고리 HYBRID 결과")
    print(f"{'='*60}")
    print(f"전체 정확도: {accuracy:.2%} ({df['acc'].sum()}/{len(df)})")
    print(f"\n카테고리별:")
    for cat in high_performance_categories:
        cat_df = df[df['gt'] == cat]
        if len(cat_df) > 0:
            cat_acc = cat_df['acc'].mean()
            cat_count = len(cat_df)
            cat_correct = cat_df['acc'].sum()
            print(f"  {cat:20s}: {cat_acc:6.1%} ({cat_correct:.0f}/{cat_count})")
    print(f"{'='*60}")
    print(f"💾 저장: {output_csv}")


if __name__ == "__main__":
    main()
