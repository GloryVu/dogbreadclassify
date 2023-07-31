import streamlit as st
import wget
import time
import zipfile
import os
st.write('Upload Dogbreeds classification dataset:')
uploaded_file = st.file_uploader("Upload Dataset .zip",type=['zip'], accept_multiple_files=False)
st.write('note: Dataset .zip must structured with pytorch format')
st.write('root')
st.write('├── class1')
st.write('│   ├── image1')
st.write('│   └── image1')
st.write('├── class2')
st.write('│   ├── image1')
st.write('│   └── image2')
st.write('│   └── image3')
if uploaded_file is not None:
    file_name = uploaded_file.name
    upload_time = time.strftime("dataset_%H_%M_%S", time.localtime())
    zip_path = 'tmp/' + uploaded_file.name
    bytes_data = uploaded_file.read()
    try:
        with open(zip_path, 'wb') as f: 
            f.write(bytes_data)
        # st.write(bytes_data)
        with zipfile.ZipFile(zip_path, "r") as zip:
            zip.extractall('datasets/'+upload_time)
        os.remove(zip_path)   
    except Exception:
        st.write('file uploaded is incorrect')
        os.remove(zip_path)