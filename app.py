import streamlit as st
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Define the model name
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

# Load pre-trained model and tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Streamlit UI components
st.title("Chatbot for Question Answering")
st.write("Ask me a question, and I will answer based on the context.")

# Input the context (you can provide a default context here or allow the user to input)
context = st.text_area("Enter the context", """
Hugging Face is a company that provides state-of-the-art Natural Language Processing technology.
It is famous for creating the Transformers library which is widely used for tasks such as text classification, question answering, and language modeling.
""")

# Input the question
question = st.text_input("Enter your question:")

# If question and context are provided, process the input and generate an answer
if question and context:
    # Tokenize the input
    inputs = tokenizer(question, context, return_tensors="pt")

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the answer
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Find the start and end positions of the answer in the context
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    # Convert token indices to actual answer words
    answer_tokens = inputs.input_ids[0][start_index:end_index + 1]
    answer = tokenizer.decode(answer_tokens)

    # Display the answer
    st.write(f"**Answer:** {answer}")
