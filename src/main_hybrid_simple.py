#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid Classification: Rule-based + Retrieval
- 키워드 규칙으로 먼저 분류
- 애매한 경우 Retrieval 기반 투표
"""

import argparse
import json
import os
import random
import re
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
    """대소문자 구분 없이 필드 찾기"""
    for k in keys:
        if k in d:
            return d[k]
    return ""

def get_label(r: Dict, label_field: str) -> str:
    """Ground truth label 추출"""
    v = None
    for key in r.keys():
        if key.lower() == label_field.lower():
            v = r[key]
            break
    
    if v is None or v == "":
        return "UNKNOWN"
    return str(v)

def read_jsonl(path: str) -> List[Dict]:
    """JSONL 로더"""
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
    """검색용 텍스트 생성"""
    headline = get_field(r, "headline", "Headline", "Title") or ""
    summary = get_field(r, "summary_en", "Summary_en", "Summary") or ""
    keywords = get_field(r, "keywords", "Keywords") or ""
    products = get_field(r, "products", "Products") or ""
    
    return f"{headline} {summary} {keywords} {products}".lower()


# -----------------------------
# Rule-based Classifier
# -----------------------------
class RuleBasedClassifier:
    """키워드 기반 규칙 분류기"""
    
    def __init__(self):
        self.rules = {
            "TV": [
                "oled", "qled", "tv", "television", "display", "screen", "webos",
                "4k", "8k", "smart tv", "gaming tv"
            ],
            "Home Appliance": [
                "washer", "dryer", "refrigerator", "fridge", "dishwasher",
                "air purifier", "vacuum", "styler", "tromm", "objet collection"
            ],
            "Vehicle/Autos": [
                "vehicle", "automotive", "car", "auto", "sdv", "motort rend",
                "vs company", "vehicle solution", "infotainment"
            ],
            "IT Product": [
                "gram", "laptop", "notebook", "computer", "monitor", "pc",
                "ultrabook", "chromebook"
            ],
            "HVAC": [
                "hvac", "air conditioning", "air conditioner", "heating",
                "ventilation", "thermostat", "multi v"
            ],
            "Audio": [
                "speaker", "soundbar", "audio", "sound", "xboom", "tone free",
                "earbuds", "headphone"
            ],
            "Advanced Tech": [
                "ai", "artificial intelligence", "robot", "thinq", "deep learning",
                "machine learning", "smart home", "iot"
            ],
            "Signage": [
                "signage", "digital signage", "commercial display", "led wall",
                "video wall"
            ],
            "Company": [
                "partnership", "collaboration", "acquisition", "investment",
                "ceo", "executive", "earnings", "revenue", "stock"
            ]
        }
    
    def classify(self, text: str) -> Dict:
        """규칙 기반 분류"""
        text = text.lower()
        
        # 각 카테고리별 매칭 점수
        scores = {}
        for category, keywords in self.rules.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[category] = score
        
        if not scores:
            return {"label": "UNKNOWN", "confidence": 0.0, "method": "rule"}
        
        # 최고 점수
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


# -----------------------------
# Retrieval-based Classifier
# -----------------------------
class RetrievalClassifier:
    """Retrieval 기반 투표 분류기"""
    
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
        """텍스트 임베딩"""
        v = self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        return v
    
    def classify(self, text: str, top_k: int = 10) -> Dict:
        """Retrieval 기반 투표"""
        qv = self.embed([text])
        scores, ids = self.index.search(qv, top_k)
        ids = ids[0].tolist()
        scores = scores[0].tolist()
        
        # 라벨 투표
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
        
        # 최다 득표
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


# -----------------------------
# Hybrid Classifier
# -----------------------------
class HybridClassifier:
    """Rule + Retrieval Hybrid"""
    
    def __init__(self, rule_classifier: RuleBasedClassifier, retrieval_classifier: RetrievalClassifier):
        self.rule = rule_classifier
        self.retrieval = retrieval_classifier
    
    def classify(self, text: str, rule_threshold: float = 0.5) -> Dict:
        """
        Hybrid 분류:
        1. Rule-based로 먼저 시도
        2. Confidence가 낮으면 Retrieval 사용
        """
        # Step 1: Rule-based
        rule_result = self.rule.classify(text)
        
        # High confidence → Rule 결과 사용
        if rule_result["confidence"] >= rule_threshold:
            return rule_result
        
        # Low confidence → Retrieval 사용
        retrieval_result = self.retrieval.classify(text)
        
        # Retrieval이 더 확신하면 사용
        if retrieval_result["confidence"] > rule_result["confidence"]:
            return retrieval_result
        
        # 둘 다 애매하면 Rule 우선
        return rule_result


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_hybrid(
    data_path: str,
    index_dir: str,
    label_field: str,
    encoder_name: str,
    n_samples: int,
    output_csv: str,
    rule_threshold: float = 0.5,
    seed: int = 42
):
    """Hybrid 평가"""
    random.seed(seed)
    np.random.seed(seed)
    
    # 데이터 로드
    rows = read_jsonl(data_path)
    all_labels = sorted(set(get_label(r, label_field) for r in rows if get_label(r, label_field) != "UNKNOWN"))
    
    print(f"\n📊 데이터 통계:")
    print(f"   전체 샘플: {len(rows)}")
    print(f"   카테고리: {len(all_labels)}")
    
    # 카테고리별 균등 샘플링
    category_samples = defaultdict(list)
    for r in rows:
        label = get_label(r, label_field)
        if label != "UNKNOWN":
            category_samples[label].append(r)
    
    samples_per_category = n_samples // len(all_labels)
    test_samples = []
    for label in all_labels:
        available = category_samples[label]
        sampled = random.sample(available, min(samples_per_category, len(available)))
        test_samples.extend(sampled)
    
    print(f"\n✅ 균등 샘플링 완료: {len(test_samples)}개")
    
    # Classifiers 초기화
    rule_clf = RuleBasedClassifier()
    retrieval_clf = RetrievalClassifier(index_dir, encoder_name)
    hybrid_clf = HybridClassifier(rule_clf, retrieval_clf)
    
    # 평가
    results = []
    for i, sample in enumerate(tqdm(test_samples, desc="Evaluating")):
        headline = get_field(sample, "headline", "Headline", "Title")
        summary = get_field(sample, "summary_en", "Summary_en", "Summary")
        keywords = get_field(sample, "keywords", "Keywords")
        products = get_field(sample, "products", "Products")
        gt_label = get_label(sample, label_field)
        
        text = f"{headline} {summary} {keywords} {products}"
        
        try:
            pred = hybrid_clf.classify(text, rule_threshold=rule_threshold)
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
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    
    # 통계
    accuracy = df['acc'].mean()
    print(f"\n{'='*60}")
    print(f"📊 HYBRID RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.2%} ({df['acc'].sum()}/{len(df)})")
    print(f"\nPer-category:")
    print(df.groupby('gt')['acc'].agg(['mean', 'count']))
    print(f"\nPer-method:")
    print(df.groupby('method')['acc'].agg(['mean', 'count']))
    print(f"{'='*60}")
    
    return df


def build_index(data_path: str, index_dir: str, label_field: str, encoder_name: str, batch_size: int = 64):
    """FAISS 인덱스 생성"""
    if faiss is None or SentenceTransformer is None:
        raise RuntimeError("faiss / sentence-transformers not installed.")

    os.makedirs(index_dir, exist_ok=True)
    rows = read_jsonl(data_path)
    if not rows:
        raise ValueError("No rows in dataset")

    texts = [build_doc_text(r) for r in rows]
    meta = pd.DataFrame([{
        "row_id": i,
        "article_id": get_field(r, "article_id", "Article_Id", "UniversalMessageId"),
        "headline": get_field(r, "headline", "Headline", "Title"),
        "summary_en": get_field(r, "summary_en", "Summary_en", "Summary"),
        "keywords": get_field(r, "keywords", "Keywords"),
        "products": get_field(r, "products", "Products"),
        "label": get_label(r, label_field),
    } for i, r in enumerate(rows)])

    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    
    model = SentenceTransformer(encoder_name, local_files_only=True)
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        v = model.encode(batch, convert_to_numpy=True, show_progress_bar=False).astype("float32")
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        embs.append(v)
    embs = np.vstack(embs).astype("float32")

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    meta.to_parquet(os.path.join(index_dir, "meta.parquet"), index=False)

    print(f"[OK] Saved: {index_dir}/faiss.index")
    print(f"[OK] Saved: {index_dir}/meta.parquet")
    print(f"[OK] Docs : {len(meta)} | dim={dim}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/A_company_202501_clean.jsonl")
    parser.add_argument("--index_dir", default="index")
    parser.add_argument("--label_field", default="Article_Category")
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--build_index", action="store_true")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--rule_threshold", type=float, default=0.5)
    parser.add_argument("--output", default="runs/results_hybrid_50.csv")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    if args.build_index:
        print("🔨 Building FAISS index...")
        build_index(args.data, args.index_dir, args.label_field, args.encoder)
    
    print("\n🚀 Starting Hybrid Evaluation...")
    evaluate_hybrid(
        args.data,
        args.index_dir,
        args.label_field,
        args.encoder,
        args.n_samples,
        args.output,
        args.rule_threshold,
        args.seed
    )


if __name__ == "__main__":
    main()
