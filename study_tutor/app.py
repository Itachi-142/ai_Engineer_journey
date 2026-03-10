import streamlit as st
import os
from dotenv import load_dotenv

os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from tutor_engine import explain_concept, socratic_chat
from quiz_generator import generate_quiz
from evaluator import evaluate_answer

st.set_page_config(page_title="AI Learning Tutor", page_icon="🎓")
st.title("🎓 AI Learning Tutor")

mode = st.radio("Choose mode:", ["Concept Explanation", "Socratic Tutor", "Quiz Me"])

st.divider()

# --- CONCEPT EXPLANATION MODE ---
if mode == "Concept Explanation":
    st.subheader("📘 Concept Explanation")
    topic = st.text_input("Enter a concept to explain:", placeholder="e.g. gradient descent, binary search, transformers")
    
    if st.button("Explain"):
        if topic.strip():
            with st.spinner("Thinking..."):
                result = explain_concept(topic)
            st.markdown(result)
        else:
            st.warning("Please enter a concept first.")

# --- SOCRATIC TUTOR MODE ---
elif mode == "Socratic Tutor":
    st.subheader("🧠 Socratic Tutor")
    st.caption("I will never give you the answer directly. I will guide you there.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "display_history" not in st.session_state:
        st.session_state.display_history = []

    for msg in st.session_state.display_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question or describe a problem...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, st.session_state.chat_history = socratic_chat(
                    user_input,
                    st.session_state.chat_history
                )
            st.markdown(response)

        st.session_state.display_history.append({"role": "user", "content": user_input})
        st.session_state.display_history.append({"role": "assistant", "content": response})

    if st.button("🔄 Reset conversation"):
        st.session_state.chat_history = []
        st.session_state.display_history = []
        st.rerun()

# --- QUIZ ME MODE ---
elif mode == "Quiz Me":
    st.subheader("📝 Quiz Me")

    topic_input = st.text_input("Enter a topic to be quizzed on:", placeholder="e.g. binary search, neural networks")

    if st.button("Generate Quiz"):
        if topic_input.strip():
            with st.spinner("Generating quiz..."):
                st.session_state.quiz_questions = generate_quiz(topic_input)
                st.session_state.current_question = 0
                st.session_state.user_answers = {}
                st.session_state.quiz_finished = False
        else:
            st.warning("Enter a topic first.")

    if "quiz_questions" in st.session_state and not st.session_state.get("quiz_finished", False):
        questions = st.session_state.quiz_questions
        idx = st.session_state.current_question

        if idx < len(questions):
            q = questions[idx]
            st.markdown(f"**Question {idx+1} of {len(questions)}**")
            st.markdown(f"**{q['question']}**")

            selected = st.radio("Choose your answer:", q["options"], key=f"q_{idx}")

            if st.button("Submit Answer"):
                st.session_state.user_answers[idx] = selected
                if idx + 1 < len(questions):
                    st.session_state.current_question += 1
                    st.rerun()
                else:
                    st.session_state.quiz_finished = True
                    st.rerun()

    if st.session_state.get("quiz_finished", False):
        st.subheader("📊 Results")
        questions = st.session_state.quiz_questions
        total_score = 0

        for i, q in enumerate(questions):
            user_ans = st.session_state.user_answers.get(i, "No answer")
            result = evaluate_answer(q["question"], q["correct_answer"], user_ans)
            total_score += result["score"]

            with st.expander(f"Q{i+1}: {q['question']}"):
                st.write(f"**Your answer:** {user_ans}")
                st.write(f"**Correct answer:** {q['correct_answer']}")
                st.write(f"**Score:** {result['score']}/5")
                st.write(f"**Feedback:** {result['feedback']}")
                st.write(f"**Explanation:** {q['explanation']}")

        avg = total_score / len(questions)
        st.metric("Final Score", f"{avg:.1f} / 5.0")

        if st.button("🔄 Start New Quiz"):
            for key in ["quiz_questions", "current_question", "user_answers", "quiz_finished"]:
                del st.session_state[key]
            st.rerun()