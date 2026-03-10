# memory.py
import json
import os
from datetime import datetime

MEMORY_FILE = "topic_memory.json"

def load_memory() -> dict:
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def save_topic(topic: str, score: float, total_questions: int):
    memory = load_memory()
    
    if topic not in memory:
        memory[topic] = []
    
    memory[topic].append({
        "score": score,
        "total_questions": total_questions,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M")
    })
    
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

def get_weak_topics(threshold: float = 3.0) -> list:
    memory = load_memory()
    weak = []
    
    for topic, attempts in memory.items():
        latest_score = attempts[-1]["score"]
        if latest_score < threshold:
            weak.append((topic, latest_score))
    
    return sorted(weak, key=lambda x: x[1])