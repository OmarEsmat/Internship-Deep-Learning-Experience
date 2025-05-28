import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, DistilBertTokenizerFast

# Load the saved tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained("notebooks/models/tokenizer_fast")
model = AutoModelForQuestionAnswering.from_pretrained("notebooks/models/model_fast")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_answer(question, context):
    """Function to generate an answer using the BERT model."""
    inputs = tokenizer(
        question, 
        context, 
        return_tensors="pt", 
        max_length=384, 
        truncation=True, 
        padding="max_length",
        return_offsets_mapping=True
    )
    offset_mapping = inputs.pop("offset_mapping")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits)
    
    sep_index = inputs["input_ids"][0].tolist().index(tokenizer.sep_token_id)
    start_index = max(start_index, sep_index + 1)
    end_index = max(end_index, sep_index + 1)
    
    if start_index > end_index:
        return "🤖 Unable to determine the answer. Try rewording your question!"
    
    answer_tokens = inputs["input_ids"][0][start_index : end_index + 1]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True)

# Streamlit App Interface
st.title("🤖 BERT Question Answering System")
st.markdown("### 🔍 Ask any question based on a given context, and BERT will find the answer!")

# Instructions
st.info("📝 **How to use:**\n"
        "1. Enter a paragraph in the **Context** box.\n"
        "2. Type your **Question** based on the context.\n"
        "3. Click **Get Answer** to see the predicted response.")

# Example button to autofill some values
if st.button("🔄 Load Example"):
    st.session_state["context"] = "The Eiffel Tower is an iron lattice tower in Paris, France. It was built in 1889 and has since become one of the most recognizable structures in the world."
    st.session_state["question"] = "Where is the Eiffel Tower located?"

# Context and Question input
context = st.text_area("📜 Enter Context:", st.session_state.get("context", ""))
question = st.text_input("❓ Enter Question:", st.session_state.get("question", ""))

# Button to get the answer
if st.button("🚀 Get Answer"):
    if context and question:
        answer = get_answer(question, context)
        st.success(f"✅ **Predicted Answer:** {answer}")
    else:
        st.warning("⚠️ Please provide both a context and a question!")

# Footer
st.markdown("---")
st.markdown("📌 **Tip:** Try using factual texts like news summaries to get the best results.")

