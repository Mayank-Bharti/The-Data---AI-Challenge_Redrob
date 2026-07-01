import json
import gzip
import pickle
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

INPUT_FILE = "datasets/candidates.jsonl"
OUTPUT_FILE = "precomputed_features.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

# HDE-CM: Fast Heuristic Filter Keywords
AI_KEYWORDS = {"llm", "rag", "embedding", "embeddings", "retrieval", "ranking", "fine-tuning", "machine learning", "pytorch", "pinecone", "vector", "generative", "nlp", "deep learning"}
TECH_TITLES = {"engineer", "developer", "data scientist", "ml", "ai", "backend", "software", "architect", "programmer", "data"}

def calculate_months_ago(date_str):
    if not date_str: return 999
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        now = datetime.now()
        return (now.year - date_obj.year) * 12 + (now.month - date_obj.month)
    except:
        return 999

def is_honeypot_or_trap(profile, skills, history):
    title = profile.get("current_title", "").lower()
    yoe = profile.get("years_of_experience", 0)
    
    is_tech_role = any(t in title for t in TECH_TITLES)
    ai_skill_count = sum(1 for s in skills if s.get("name", "").lower() in AI_KEYWORDS)
    
    # Keyword Stuffer Trap
    if not is_tech_role and ai_skill_count > 3:
        return True

    # Chronological Trap
    total_history_months = sum(h.get("duration_months", 0) for h in history)
    history_yoe = total_history_months / 12.0
    
    if yoe > 0 and history_yoe > 0:
        if yoe > history_yoe * 2.5 or history_yoe > yoe * 2.5:
            return True

    return False

def fast_pre_filter(cand):
    """
    HDE-CM Layer 1: Fast filtering to reduce 100K candidates to ~5-10K.
    Saves massive compute time on neural embeddings.
    """
    profile = cand.get("profile", {})
    skills = cand.get("skills", [])
    signals = cand.get("redrob_signals", {})
    
    yoe = profile.get("years_of_experience", 0)
    if yoe < 2 or yoe > 16:
        return False
        
    months_inactive = calculate_months_ago(signals.get("last_active_date", ""))
    if months_inactive > 12:
        return False # Truly dead account
        
    # Must have some tech relevance
    title = profile.get("current_title", "").lower()
    is_tech_role = any(t in title for t in TECH_TITLES)
    has_ai_skill = any(s.get("name", "").lower() in AI_KEYWORDS for s in skills)
    
    if not is_tech_role and not has_ai_skill:
        return False
        
    return True

def extract_features(cand):
    profile = cand.get("profile", {})
    history = cand.get("career_history", [])
    skills = cand.get("skills", [])
    signals = cand.get("redrob_signals", {})
    
    is_trap = is_honeypot_or_trap(profile, skills, history)
    
    # Text for semantic embedding
    text_corpus = (
        profile.get("headline", "") + " " + 
        profile.get("summary", "") + " " + 
        " ".join([h.get("title", "") for h in history]) + " " +
        " ".join([h.get("description", "") for h in history])
    ).lower()
    
    # Base elements
    yoe = profile.get("years_of_experience", 0)
    loc = profile.get("location", "").lower()
    will_relocate = signals.get("willing_to_relocate", False)
    
    # CM: Career Momentum Calculation
    months_inactive = calculate_months_ago(signals.get("last_active_date", ""))
    resp_rate = signals.get("recruiter_response_rate", 0.5)
    gh_score = signals.get("github_activity_score", -1)
    apps_30d = signals.get("applications_submitted_30d", 0)
    
    return {
        "id": cand.get("candidate_id"),
        "is_trap": is_trap,
        "text_corpus": text_corpus,
        "yoe": yoe,
        "loc": loc,
        "will_relocate": will_relocate,
        "months_inactive": months_inactive,
        "resp_rate": resp_rate,
        "gh_score": gh_score,
        "apps_30d": apps_30d,
        "title": profile.get("current_title", "Engineer"),
        "notice_period_days": signals.get("notice_period_days", 30)
    }

def main():
    print("Parsing candidates with HDE-CM fast filter...")
    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} not found!")
        return
        
    open_func = gzip.open if INPUT_FILE.endswith(".gz") else open
    features_list = []
    
    with open_func(INPUT_FILE, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Filtering"):
            line = line.strip()
            if not line: continue
            cand = json.loads(line)
            
            if fast_pre_filter(cand):
                features = extract_features(cand)
                if not features["is_trap"]:
                    features_list.append(features)
            
    print(f"Filtered down to {len(features_list)} highly viable candidates.")
    
    print("Loading HDE Sentence Transformer model...")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"Embedding {len(features_list)} candidates...")
    texts = [f["text_corpus"] for f in features_list]
    
    # Batch encode
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
    
    for i, f in enumerate(features_list):
        f["embedding"] = embeddings[i]
        del f["text_corpus"] # remove raw text to save space
        
    print(f"Saving HDE-CM vectors to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(features_list, f)
        
    print("Pre-computation complete! The 5-minute sandbox is now ready.")

if __name__ == "__main__":
    main()
