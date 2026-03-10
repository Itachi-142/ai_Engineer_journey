# 🎓 AI Learning Tutor

A personalized AI-powered learning assistant that explains concepts, guides you through problems using the Socratic method, and quizzes you on any topic — with score tracking across sessions.

**🔗 Live Demo: [v-ai-tutor.streamlit.app](https://v-ai-tutor.streamlit.app)**

---

## Features

### 📘 Concept Explanation
Enter any topic and get a structured explanation with a simple breakdown, a real-world analogy, and a concrete example.

### 🧠 Socratic Tutor
Describe a problem you're stuck on. The tutor never gives you the answer directly — it guides you there through leading questions and hints, one step at a time.

### 📝 Quiz Me
Enter any topic and get a 5-question multiple choice quiz generated on the spot. Each answer is evaluated by an LLM judge that scores your response from 1–5 and explains why.

### 📚 Study History
The sidebar tracks every topic you've studied and your latest quiz score. Topics where you scored below 3.0 are flagged under **Needs Review** so you know what to revisit.

---

## Tech Stack

- **LLM** — LLaMA 3.3 70B via Groq API
- **Framework** — LangChain + LangChain-Groq
- **UI** — Streamlit
- **Evaluation** — LLM-as-Judge pattern
- **Memory** — JSON-based local topic tracker

---

## Project Structure

```
study_tutor/
├── app.py              # Streamlit UI
├── tutor_engine.py     # Concept explanation + Socratic chat
├── quiz_generator.py   # LLM-powered quiz generation
├── evaluator.py        # LLM-as-Judge answer scoring
├── memory.py           # Topic history + weak topic detection
└── requirements.txt
```

---

## Run Locally

```bash
git clone https://github.com/Itachi-142/ai_Engineer_journey
cd ai_Engineer_journey/study_tutor
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```

Run the app:
```bash
streamlit run app.py
```

---

Built as part of a 4-month AI Engineer learning roadmap.