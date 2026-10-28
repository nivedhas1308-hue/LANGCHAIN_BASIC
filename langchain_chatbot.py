
import csv
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

BASE = Path(__file__).parent
DATASET = BASE / "dataset"
DATASET.mkdir(exist_ok=True)

llm = ChatOllama(model="llama3.2")

# Five prompt templates for different college scenarios
templates = [
    (1, "Exam Preparation",
     "Give practical exam preparation tips for: {question}"),
    (2, "Admissions",
     "Explain the college admission process for: {question}"),
    (3, "Library",
     "Help a student with this library question: {question}"),
    (4, "Study Tips",
     "Suggest effective study habits for: {question}"),
    (5, "Fees",
     "Explain how to find official college fee information about: {question}")
]

with open(DATASET / "prompts.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["prompt_id", "scenario", "prompt_template"])
    writer.writerows(templates)

# Prepare 20 user queries
queries = [
    "How can I prepare for semester exams?",
    "What documents are needed for college admission?",
    "How can I borrow a library book?",
    "How can I improve my study habits?",
    "Where can I check college fees?",
    "How do I create a revision timetable?",
    "What should I bring for admission?",
    "How do I renew a library book?",
    "How can I avoid study distractions?",
    "Who should I contact about fee details?",
    "How can I manage exam stress?",
    "When does college admission begin?",
    "How can I find books in the library?",
    "How do I take useful class notes?",
    "Can I pay college fees online?",
    "How should I prepare for a practical exam?",
    "How can I check my application status?",
    "What are library opening hours?",
    "How can I remember what I study?",
    "Where can I find scholarship information?"
]

with open(DATASET / "user_queries.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["query_id", "query"])
    for i, q in enumerate(queries, 1):
        writer.writerow([i, q])

# Improved prompt for clearer and safer responses
improved_prompt = PromptTemplate.from_template(
    """You are a helpful College Assistant.
Answer the student's question clearly and simply.
Give 3 practical steps when suitable.
Never invent official college dates, fees, or rules.
If official information is unknown, direct the student
to the college office or official website.

Student question: {question}
Answer:"""
)

# Two additional chains: summarization and question answering
summary_prompt = PromptTemplate.from_template(
    "Summarize the following text in 3 short bullet points:\n{text}"
)
summary_chain = summary_prompt | llm

qa_prompt = PromptTemplate.from_template(
    "Answer the question using only the context below. "
    "If the answer is not in the context, say so.\n"
    "Context: {context}\nQuestion: {question}\nAnswer:"
)
qa_chain = qa_prompt | llm

# Run all 20 queries and save responses
response_rows = []
comparison_rows = []

for i, question in enumerate(queries, 1):
    scenario = templates[(i - 1) % len(templates)][1]

    old_prompt = PromptTemplate.from_template(
        "You are a college assistant. Answer this question: {question}"
    )
    old_answer = (old_prompt | llm).invoke(
        {"question": question}
    ).content

    new_answer = improved_prompt.invoke(
        {"question": question}
    )
    new_answer = llm.invoke(new_answer).content

    response_rows.append([
        i, scenario, question, new_answer
    ])

    if i == 1:
        comparison_rows.append([
            question, old_answer, new_answer
        ])

    print(f"\nQuery {i}: {question}")
    print("Assistant:", new_answer)

with open(DATASET / "chatbot_responses.csv", "w",
          newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["query_id", "scenario", "query", "response"])
    writer.writerows(response_rows)

# Run summarization and question-answering chains
summary = summary_chain.invoke({
    "text": "Students should create a study timetable, review notes "
            "regularly, practice sample questions, take breaks, "
            "and sleep well before exams."
}).content

qa_answer = qa_chain.invoke({
    "context": "The college library opens at 9 AM on weekdays.",
    "question": "When does the college library open on weekdays?"
}).content

with open(DATASET / "chain_results.csv", "w",
          newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["task", "input", "output"])
    writer.writerow(["Prompt comparison - original",
                     queries[0], comparison_rows[0][1]])
    writer.writerow(["Prompt comparison - improved",
                     queries[0], comparison_rows[0][2]])
    writer.writerow(["Summarization", "Study preparation text", summary])
    writer.writerow(["Question Answering",
                     "Weekday library opening time", qa_answer])

print("\nAll 20 queries processed.")
print("CSV results saved in the dataset folder.")
print("Summarization and question-answering chains completed.")
