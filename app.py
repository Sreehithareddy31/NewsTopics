import streamlit as st
import numpy as np
from transformers import BertTokenizer,BertForSequenceClassification
import torch

print("Streamlit app is starting...")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model=BertForSequenceClassification.from_pretrained("Sreehitha31/bbcnews",num_labels=5)

def predict_bbc_category(text):
    inputs=tokenizer(text,return_tensors='pt')
    outputs=model(**inputs)
    logits=outputs.logits
    probs=torch.softmax(logits,dim=1).detach().numpy() 
    return probs

def main():
    print("main function")
    st.title("News Classification")
    st.write("Enter an article text")
    user_input=st.text_area("Input your text here:")
    if st.button("Classify"):
        st.write(f"Input text: {user_input}")
        probabilities=predict_bbc_category(user_input)
        st.write(f"Probabilities: {probabilities}")
        categories=['business','entertainment','politics','sport','technology']
        for i, category in enumerate(categories):
            st.write(f"Probability of {category}: {probabilities[0][i]}")
        max_prob_idx = np.argmax(probabilities, axis=1)[0]
        most_probable_category = categories[max_prob_idx]
        st.write(f"Most probable category: {most_probable_category}")
    else:
        st.write("Please input some text")

if __name__=='__main__':
    main()
