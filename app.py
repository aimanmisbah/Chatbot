# Step 2: Import Necessary Libraries
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Step 3: Load a Pre-Trained Q&A Model
model_name = "deepset/roberta-base-squad2"  # A Q&A model available on Hugging Face
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 4: Create a Q&A Pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Step 5: Set the Expanded Context with Detailed Knowledge
context = """
Cancer is a disease where abnormal cells grow uncontrollably. There are many types of cancer, each with its own specific symptoms, causes, and treatment options.

Types of Cancer:
- *Breast Cancer*: Common symptoms include lumps, changes in breast shape, and skin dimpling. Treatments include surgery, radiation, and chemotherapy.
- *Lung Cancer*: Symptoms include coughing, chest pain, and weight loss. Risk factors include smoking, environmental toxins, and genetic factors.
- *Colorectal Cancer*: Symptoms include changes in bowel habits, blood in stool, and abdominal discomfort. Early detection through screening improves survival rates.
- *Prostate Cancer*: Common in older men. Symptoms include difficulty urinating, pelvic discomfort, and bone pain. Treatment may include radiation, surgery, and hormone therapy.

Cancer Stages and Treatments:
- *Stage 1 & 2*: Often involves surgery and radiation for localized tumors.
- *Stage 3*: More advanced and may require chemotherapy and radiation.
- *Stage 4*: Involves metastasis and may include targeted therapy, immunotherapy, and palliative care.

Diet and Nutrition:
- Cancer patients are advised to consume a high-protein diet to maintain strength.
- Fruits and vegetables rich in antioxidants, like berries and leafy greens, support immune health.
- Avoid processed foods, sugary drinks, and excessive red meat.

Prevention Tips:
- *Quit Smoking*: Reduces the risk of lung and several other cancers.
- *Healthy Diet*: Focus on whole foods, limit alcohol, and include fiber for colon health.
- *Regular Screenings*: Early detection increases treatment success.

This context is continually updated with the latest cancer research, innovative treatments, and lifestyle recommendations for prevention and support.
"""

# Step 6: Define a Function to Ask Questions
def ask_question(question):
    result = qa_pipeline(question=question, context=context)
    return result["answer"]

# Step 7: Start Asking Questions
while True:
    question = input("Ask a question about cancer (type 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    answer = ask_question(question)
    print("Bot:", answer)
    print("-" * 50)
