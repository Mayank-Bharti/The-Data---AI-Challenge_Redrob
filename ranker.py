import csv
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import time

OUTPUT_FILE = "The_Crimson_Codex.csv"
INPUT_FEATURES = "precomputed_features.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

# HDE Axis: The specific Job Description text vector representation
JD_TEXT = """
Deep technical depth in modern ML systems embeddings, retrieval, ranking, LLMs, fine-tuning.
Production experience with embeddings-based retrieval systems deployed to real users. 
Production experience with vector databases or hybrid search infrastructure.
Strong Python.
Hands-on experience designing evaluation frameworks for ranking systems NDCG, MRR, MAP.
"""

def generate_hde_cm_reasoning(cand, sim_score, final_score):
    """XAI (Explainable AI) Reasoning Generator"""
    reasons = []
    
    # 1. Semantic Match
    if sim_score > 0.4:
        reasons.append(f"Strong semantic fit ({sim_score:.2f}) to retrieval/ranking systems")
    elif sim_score > 0.2:
        reasons.append(f"Solid NLP/ML semantic match ({sim_score:.2f})")
        
    # 2. Chronological/YoE
    yoe = cand["yoe"]
    if 5 <= yoe <= 9:
        reasons.append(f"Ideal {yoe} YoE in applied ML")
    elif yoe > 9:
        reasons.append(f"Highly senior with {yoe} YoE")
    else:
        reasons.append(f"{yoe} YoE")
        
    # 3. Behavioral Momentum
    if cand["apps_30d"] > 5 or cand["resp_rate"] > 0.7:
        reasons.append("High Hunt-Mode momentum")
        
    if cand["gh_score"] > 50:
        reasons.append("Strong open-source/GitHub activity")
        
    if cand["months_inactive"] > 3:
        reasons.append("Warning: Passive candidate (inactive)")
        
    # Combine
    text = f"{cand['title']}. " + "; ".join(reasons) + "."
    return text.strip()

def main():
    start_time = time.time()
    print("Initializing HDE-CM Online Engine...")
    model = SentenceTransformer(MODEL_NAME)
    jd_embedding = model.encode(JD_TEXT, convert_to_numpy=True)
    
    print(f"Loading precomputed HDE vectors from {INPUT_FEATURES}...")
    try:
        with open(INPUT_FEATURES, "rb") as f:
            candidates = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FEATURES} not found. Ensure precompute.py is run first.")
        return
        
    print(f"Executing CM-Ranking for {len(candidates)} shortlisted candidates...")
    
    scored_candidates = []
    
    for cand in candidates:
        # HDE: Semantic Distance (Cosine Similarity)
        cand_emb = cand["embedding"]
        sim_score = np.dot(jd_embedding, cand_emb) / (np.linalg.norm(jd_embedding) * np.linalg.norm(cand_emb))
        
        # Base Score mapped 0-100
        base_score = float(sim_score) * 100 
        
        # Heuristics
        yoe = cand["yoe"]
        if 5 <= yoe <= 9:
            base_score += 20
        elif 3 <= yoe < 5 or 9 < yoe <= 12:
            base_score += 10
            
        loc = cand["loc"]
        if "pune" in loc or "noida" in loc:
            base_score += 15
        elif cand["will_relocate"]:
            base_score += 5
            
        # CM: Career Momentum Layer
        momentum_multiplier = 1.0
        
        # 1. Recruiter Response (0.5 to 1.5 multiplier)
        momentum_multiplier *= (0.5 + cand["resp_rate"])
        
        # 2. Activity (Hunt Mode vs Passive Decay)
        if cand["months_inactive"] > 6:
            momentum_multiplier *= 0.2  # Massive exponential decay
        elif cand["months_inactive"] <= 1 and cand["apps_30d"] > 0:
            momentum_multiplier *= 1.2  # Active hunter
            
        # 3. Open Source
        if cand["gh_score"] > 50:
            momentum_multiplier *= 1.15
            
        final_score = base_score * momentum_multiplier
        
        cand["score"] = final_score
        cand["sim_score"] = float(sim_score)
        
        # Explainable AI (XAI) output
        cand["reasoning"] = generate_hde_cm_reasoning(cand, sim_score, final_score)
        
        scored_candidates.append(cand)
        
    # Sort and rank
    print("Finalizing Top 100 Leaderboard...")
    for cand in scored_candidates:
        cand["score"] = round(cand["score"], 6)
        
    scored_candidates.sort(key=lambda x: (-x["score"], x["id"]))
    top_100 = scored_candidates[:100]
    
    print(f"Writing output to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for rank, cand in enumerate(top_100, start=1):
            writer.writerow([cand["id"], rank, f"{cand['score']:.4f}", cand["reasoning"]])
            
    elapsed = time.time() - start_time
    print(f"Done! Top 100 evaluated in {elapsed:.2f} seconds. (HDE-CM architecture strict constraint pass).")

if __name__ == "__main__":
    main()