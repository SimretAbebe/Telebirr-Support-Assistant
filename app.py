import os
if not os.path.exists("./chroma_db"):
    os.system("python src/chroma_setup.py")
    os.system("python src/chroma_setup_am.py")

import gradio as gr
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from retriever import retrieve
from generator import generate_answer

def answer_question(question, language, num_sources):
    if not question.strip():
        return "Please enter a question.", ""

    greetings = ["hello", "hi", "hey", "help", "how can u help", "how can you help", "what can you do"]
    q_lower = question.strip().lower()
    if q_lower in greetings or (len(q_lower) < 20 and any(g in q_lower for g in greetings)):
        intro = ("Hi! I can help answer questions about telebirr — sending money, cash-out at agents, "
                 "resetting your PIN, failed transactions, and more. What would you like to know?")
        return intro, ""

    lang_code = "am" if language == "Amharic" else "en"
    matches = retrieve(question, language=lang_code, top_k=int(num_sources))

    combined_context = ""
    sources_display = ""
    for i, (score, item) in enumerate(matches, start=1):
        combined_context += f"Q: {item['question']}\nA: {item['answer']}\n\n"
        sources_display += f"### Source {i}\n"
        sources_display += f"- **Matched question:** {item['question']}\n"
        sources_display += f"- **Confidence:** {score:.2f}\n\n"

    answer = generate_answer(question, combined_context, language=lang_code)
    return answer, sources_display


with gr.Blocks(title="Telebirr Support Assistant") as demo:
    gr.Markdown("# Telebirr Support Assistant")
    gr.Markdown("Ask plain-language questions about telebirr — sending money, cash-out, failed transactions, and more. Grounded in real, documented answers only.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Settings")
            language = gr.Radio(
                choices=["English", "Amharic"],
                value="English",
                label="Language",
                info="Choose which language to ask in"
            )
            num_sources = gr.Slider(
                minimum=1, maximum=5, value=3, step=1,
                label="Number of Sources",
                info="How many knowledge base entries to retrieve"
            )

        with gr.Column(scale=2):
            gr.Markdown("## Ask a Question")
            question = gr.Textbox(
                label="Your Question",
                placeholder="e.g. How do I send money?",
                lines=3
            )
            with gr.Row():
                ask_btn = gr.Button("Ask Question", variant="primary")
                clear_btn = gr.Button("Clear")

            gr.Markdown("## Answer")
            answer_output = gr.Markdown()

            gr.Markdown("## Sources Used")
            sources_output = gr.Markdown()

    ask_btn.click(
        fn=answer_question,
        inputs=[question, language, num_sources],
        outputs=[answer_output, sources_output]
    )
    clear_btn.click(
        fn=lambda: ("", "English", 3, "", ""),
        outputs=[question, language, num_sources, answer_output, sources_output]
    )

if __name__ == "__main__":
    demo.launch()