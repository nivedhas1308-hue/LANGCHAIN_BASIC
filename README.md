Week 10 - LangChain Basics

Project Title
College Assistant Chatbot Using LangChain and Ollama

Objective
To develop a domain-specific chatbot using LangChain and test
five prompt scenarios with at least 20 user queries.

Technologies Used
- Python
- LangChain
- Ollama
- Llama 3.2
- Jupyter Notebook

Main Features
1. Five college-related prompt scenarios.
2. Testing with 20 user queries.
3. Comparison of original and improved prompts.
4. Summarization chain.
5. Question-answering chain.
6. CSV files containing prompts, queries, responses, and chain results.

How to Run
Install the required packages:

pip install -r requirements.txt

Ensure Ollama is running and the llama3.2 model is available.

Run:

python langchain_chatbot.py

Week 9 and Week 10 Comparison
Week 9 used a basic Ollama chatbot with a direct interaction loop.
Week 10 uses LangChain prompt templates and chains to organize
the chatbot workflow and support additional tasks.

Observations
The chatbot generated answers to college-related questions.
The improved prompt provides clearer instructions and reminds
the model not to invent official college information.
The summarization and question-answering chains perform separate tasks.

Output Files
- dataset/prompts.csv
- dataset/user_queries.csv
- dataset/chatbot_responses.csv
- dataset/chain_results.csv

Notebook
notebook/Week10_LangChain_Basics.ipynb


## Prompt Improvement and Output Comparison

The original prompt provided general instructions to answer
student questions. The improved prompt instructed the chatbot
to use simple language, provide practical steps, and avoid
inventing official college information.

The improved prompt aims to produce clearer and more reliable
answers. The outputs are saved in dataset/chain_results.csv
for comparison.

## Conclusion

The College Assistant chatbot was developed using LangChain
and Ollama with the Llama 3.2 model. Five prompt scenarios
were designed and 20 user queries were tested. Additional
chains were created for summarization and question answering.
The project demonstrates how LangChain can organize prompts
and support multiple chatbot tasks.


