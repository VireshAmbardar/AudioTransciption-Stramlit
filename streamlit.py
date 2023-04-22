import whisper
import streamlit as st
import numpy as np
import io

## pip install streamlit pyinstaller
## Whisper module


transcription = ''
st.title("Transcribe Audio")

# to remove menu button
st.markdown("""
<style>
    .css-1rs6os.edgvbvh3
        {
            visibility:hidden;
        }
    
    .css-z3au9t.egzxvld2 
        {
            visibility:hidden;
        }
    .css-cio0dv.egzxvld1
        {
            visibility:hidden;
        }
</style>
""", unsafe_allow_html=True)



## radio button
option = st.radio(label="Select your file type", options=["audio", "video"] )

# Increase the maximum upload size to 500 MB
# st.set_option('server.max_request_size', 1000 * 1024 * 1024)


## audio option
if option == "audio":

    audio_file = st.file_uploader("**Upload Audio**", type=["mp3", "wav", 'm4a'], accept_multiple_files=False)
    if audio_file:
        st.write("Transcription started... Please be patient")

        ## load the model and get trancription
        model = whisper.load_model("small", device="cpu")
        st.write("Please be patient")
        transcription = model.transcribe(audio_file.name, language="English", fp16=False)
        st.text_area("Here is your Transcriptions", value=transcription["text"] ,  disabled=True)
        # st.text()

        st.download_button(label="download", data=transcription["text"], file_name="transcription.txt")

## video option
if option == "video":

    audio_file = st.file_uploader("**Upload Audio**", type=["mp4", "avi", 'wmv',"mkv","mov"], accept_multiple_files=False)
    if audio_file:
        st.write("Transcription started... Please be patient")

        ## load the model and get trancription
        model = whisper.load_model("small", device="cpu")
        st.write("Please be patient")
        transcription = model.transcribe(audio_file.name, language="English", fp16=False)
        st.text_area("Here is your Transcriptions", value=transcription["text"] ,  disabled=True)
        # st.text()

        st.download_button(label="download", data=transcription["text"], file_name="transcription.txt")
    

    ## to upload size of limit 1 gb run "streamlit run streamlit.py --server.maxUploadSize 1024" in terminal