import streamlit as st
import numpy as np
import nltk
from keras.models import load_model, model_from_json
nltk.download('punkt')
import json

# Paths for models (user 1)
path_enc_model_u1 = "Model/encoder_model.json"
path_dec_model_u1 = "Model/decoder_model.json"

# Paths for weights (user 1)
path_enc_weights_u1 = "Model/encoder_weights.h5"
path_dec_weights_u1 = "Model/decoder_weights.h5"

# Paths for pickle files (user 1)
ipl_u1 = "Data/input_lang.json"
opl_u1 = "Data/target_lang.json"
opl1_u1 = "Data/target_lang1.json"

# Paths for models (user 2)
path_enc_model_u2 = "Model/encoder_model.json"
path_dec_model_u2 = "Model/decoder_model.json"

# Paths for weights (user 2)
path_enc_weights_u2 = "Model/encoder_weights.h5"
path_dec_weights_u2 = "Model/decoder_weights.h5"

# Paths for pickle files (user 2)
ipl_u2 = "Data/input_lang.json"
opl_u2 = "Data/target_lang.json"
opl1_u2 = "Data/target_lang1.json"

# Load dictionaries
def load_data(ip_lang,op_lang,op_lang1):
    
    with open(ip_lang, 'rb') as input:
        input_lang = json.load(input)
    
    with open(op_lang, 'rb') as target:
        target_lang = json.load(target)

    with open(op_lang1, 'rb') as target1:
        target_lang1 = json.load(target1)
    
    input.close()
    target.close()
    target1.close()
    
    return input_lang,target_lang,target_lang1

# Coverting sentence into vectors
def sentence_to_vector(sentence, lang,len_input):
    
    pre = sentence
    vec = np.zeros(len_input)
    sentence_list = [lang[s] for s in pre.split(' ')]
    for i,w in enumerate(sentence_list):
        vec[i] = w
    return vec

# Given an input string, an encoder model (infenc_model) and a decoder model (infmodel),
def translate(input_sentence, infenc_model, infmodel,path_ipl, path_opl,path_opl1,len_input,len_target):
    
    # # Load input and output lang
    input_lang, target_lang, target_lang1 = load_data(path_ipl, path_opl,path_opl1)

    # calling sentence to vector function
    sv = sentence_to_vector(input_sentence, input_lang,len_input)
    sv = sv.reshape(1,len(sv))
    [emb_out, sh, sc] = infenc_model.predict(x=sv)
    
    i = 0
    start_vec = target_lang["<start>"]
    stop_vec = target_lang["<end>"]
    
    cur_vec = np.zeros((1,1))
    cur_vec[0,0] = start_vec
    cur_word = "<start>"
    output_sentence = ""

    while cur_word != "<end>" and i < (len_target-1):
        i += 1
        if cur_word != "<start>":
            output_sentence = output_sentence + " " + cur_word
        x_in = [cur_vec, sh, sc]
        [nvec, sh, sc] = infmodel.predict(x=x_in)
        cur_vec[0,0] = np.argmax(nvec[0,0])
        cur_word = target_lang1[str(np.argmax(nvec[0,0]))]
    output_sentence = input_sentence + " " + output_sentence
    return output_sentence

def load_model(enc_m,dec_m,enc_w,dec_w):

    # 1. load json and create model
    with open(dec_m, "rb") as dec_json_file:
        loaded_dec_model_json = dec_json_file.read()
        loaded_dec_model = model_from_json(loaded_dec_model_json)

    with open(enc_m, "rb") as enc_json_file:
        loaded_enc_model_json = enc_json_file.read()
        loaded_enc_model = model_from_json(loaded_enc_model_json)

    enc_json_file.close()
    dec_json_file.close()

    # 2.load weights into new model
    loaded_dec_model.load_weights(dec_w)
    loaded_enc_model.load_weights(enc_w)

    return loaded_dec_model,loaded_enc_model

def main():
    
    # Sidebar with 4 menu items
    rad = st.sidebar.radio("Navigation",['Home','Web App (User-1)', 'Web App (User-2)','About Us'])
    
    # Home menu
    if rad == "Home":
        st.title("CDAC Project (Group 10)")

        html_temp = """
                <div style="background-color:tomato;padding:10px">
                    <h2 style="color:white;text-align:center;">Welcome !</h2>
                </div> <br>
            """
        st.markdown(html_temp,unsafe_allow_html=True)
    
    # Web app user 1 menu
    if rad == "Web App (User-1)":
        
        # Load enc and dec models
        dec_model, enc_model = load_model(path_enc_model_u1,path_dec_model_u1,path_enc_weights_u1,path_dec_weights_u1)
        
        # HTML template
        html_temp = """
            <div style="background-color:tomato;padding:10px">
                <h2 style="color:white;text-align:center;">Personalized Sentence Autocompletion App</h2>
            </div> <br>

            <h4 style="margin-top: 10px; border: 2px solid tomato; padding: 10px; border-radius: 10px; text-align: center;">User-1</h4>
            <br>
        """
        st.markdown(html_temp,unsafe_allow_html=True)

        # Steamlit text input
        user_text = st.text_input("Enter your sentence:",placeholder="Type here")

        # Steamlit button
        if st.button("Predict"):
            if user_text != "":
                seed = translate(user_text.lower(),enc_model,dec_model,ipl_u1,opl_u1,opl1_u1,len_input_u1,len_target_u1)
                st.success(seed) # Steamlit success message
            else:
                st.error("Error! Found empty string.") # Steamlit error message

    # Web app user 2 menu
    if rad == "Web App (User-2)":
        # Load enc and dec models
        dec_model, enc_model = load_model(path_enc_model_u2,path_dec_model_u2,path_enc_weights_u2,path_dec_weights_u2)
        
        html_temp = """
            <div style="background-color:tomato;padding:10px">
                <h2 style="color:white;text-align:center;">Personalized Sentence Autocompletion App</h2>
            </div> <br>

            <h4 style="margin-top: 10px; border: 2px solid tomato; padding: 10px; border-radius: 10px; text-align: center;">User-2</h4>
            <br>
        """
        st.markdown(html_temp,unsafe_allow_html=True)

        user_text = st.text_input("Enter your sentence:",placeholder="Type here")

        if st.button("Predict"):
            if user_text != "":
                seed = translate(user_text.lower(),enc_model,dec_model,ipl_u2,opl_u2,opl1_u2,len_input_u2,len_target_u2)
                st.success(seed) # Steamlit success message
            else:
                st.error("Error! Found empty string.") # Steamlit error message
    
    # About us menu
    if rad == "About Us":
        html_temp = """
            <div style="background-color:tomato;padding:10px">
                <h2 style="color:white;text-align:center;">About Us</h2>
            </div> <br>

            <div style="margin-top: 20px;">
                <h4 style="border-left: 4px solid tomato; padding-left: 10px;">A little bit about our project</h4>
                <p style="margin-top: 10px;">We built a personalized model which autocompletes the whole sentence using NLP and tensorflow keras(BiLSTM Encoder and a LSTM Decoder) in real time. We have used Python programming language and Streamlit, HTML and CSS for Front-end of our web app.<p>
            </div>
        """
        st.markdown(html_temp,unsafe_allow_html=True)

if __name__ == "__main__":
    len_input_u1 = 31
    len_target_u1 = 37
    len_input_u2 = 31
    len_target_u2 = 37
    main()