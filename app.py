# Necessary libraries & Imports
from itertools import zip_longest
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import requests
from PIL import Image
import io
import openai
import requests
import toml
from docx import Document
from datetime import datetime
import os
from pathlib import Path
# Import your HTML templates and CSS
from htmlTemplates import bot_template, user_template, css

# Set Streamlit page configuration and API keys
st.set_page_config(page_title="✨AI Mentor and Image Generator✨")
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Main Title of the app
st.title("🤖✨ Tosief's AI Maestro 🖼️🗣️")
st.markdown("## Chat, Visualize, Speak and Transcribe")

# Initialize session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []  # Store AI generated responses

if 'past' not in st.session_state:
    st.session_state['past'] = []  # Store past user inputs

if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ""  # Initialize the latest user input

# Initialize the ChatOpenAI model for the chatbot
chat = ChatOpenAI(
    temperature=0.5,
    model_name="gpt-3.5-turbo",
    openai_api_key=openai_api_key,
    max_tokens=100
)

# Inject custom CSS for chat messages
st.markdown(css, unsafe_allow_html=True)

# Function to display messages using the HTML templates
def display_message(message, is_user=True):
    if is_user:
        # If it's a user message, use the user template
        html = user_template.replace("{{MSG}}", message)
    else:
        # If it's a bot message, use the bot template
        html = bot_template.replace("{{MSG}}", message)
    st.markdown(html, unsafe_allow_html=True)

# Function for image generation
def generate_image(text):
    #with st.spinner('Generating image...'):
        response = openai.Image.create(prompt=text, n=1, size="512x512")
        image_url = response.data[0]['url']
        image_content = requests.get(image_url).content
        image = Image.open(io.BytesIO(image_content))
        st.image(image, caption='Generated Image')

def build_message_list():
    """
    Build a list of messages including system, human and AI messages.
    """
    # Start zipped_messages with the SystemMessage
    zipped_messages = [SystemMessage(
        # content="You are a helpful AI assistant talking with a human. If you do not know an answer, just say 'I don't know', do not make up an answer.")]
        content = """your name is AI Mentor. You are an AI Technical Expert for Artificial Intelligence, here to guide and assist students with their AI-related questions and concerns. Please provide accurate and helpful information, and always maintain a polite and professional tone.

                1. Greet the user politely ask user name and ask how you can assist them with AI-related queries.
                2. Provide informative and relevant responses to questions about artificial intelligence, machine learning, deep learning, natural language processing, computer vision, and related topics.
                3. you must Avoid discussing sensitive, offensive, or harmful content. Refrain from engaging in any form of discrimination, harassment, or inappropriate behavior.
                4. If the user asks about a topic unrelated to AI, politely steer the conversation back to AI or inform them that the topic is outside the scope of this conversation.
                5. Be patient and considerate when responding to user queries, and provide clear explanations.
                6. If the user expresses gratitude or indicates the end of the conversation, respond with a polite farewell.
                7. Do Not generate the long paragarphs in response. Maximum Words should be 100.

                Remember, your primary goal is to assist and educate students in the field of Artificial Intelligence. Always prioritize their learning experience and well-being."""
    )]

    # Zip together the past and generated messages
    for human_msg, ai_msg in zip_longest(st.session_state['past'], st.session_state['generated']):
        if human_msg is not None:
            zipped_messages.append(HumanMessage(
                content=human_msg))  # Add user messages
        if ai_msg is not None:
            zipped_messages.append(
                AIMessage(content=ai_msg))  # Add AI messages

    return zipped_messages

def generate_response():
    """
    Generate AI response using the ChatOpenAI model.
    """
    try:                                         #1
        # Build the list of messages
        zipped_messages = build_message_list()

        # Generate response using the chat model
        ai_response = chat(zipped_messages)

        return ai_response.content
    except Exception as e:                       #2
        st.write(f"Error: {e}")                  #3
        return "Error generating response"       #4

# Define function to submit user input
def submit(user_input):
    # Set entered_prompt to the current value of user_input
    st.session_state.entered_prompt = user_input

# Define the callback function for clearing text
def clear_text():
    # Clear the text input in the session state
    st.session_state.user_input = ''

# Ensure that the text input uses the session state to retain its value
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ''

# Function for audio transcription
def transcribe_audio(file, language):
    # Ensure the API key is set from Streamlit secrets
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    
    try:
        # Attempt transcription using the Whisper model
        transcript_response = openai.Audio.transcribe(
            model="whisper-1",
            file=file,
            language=language
        )
        return transcript_response["text"]
    except Exception as e:
        st.error(f"Error in transcription: {e}")
        return None

def save_transcription_to_docx(transcription_text):
    try:
        # Create a new Document
        doc = Document()
        doc.add_paragraph(transcription_text)
        
        # Generate a unique filename with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"transcription_{timestamp}.docx"
        
        # Save the document in the current directory
        doc_path = os.path.join(os.getcwd(), filename)
        doc.save(doc_path)
        
        return doc_path
    except Exception as e:
        st.error(f"Failed to save document: {e}")
        return None

# Function for text-to-speech conversion
def convert_text_to_speech(text):
    # Load API key from Streamlit secrets
    api_key = st.secrets["OPENAI_API_KEY"]

    # Specify the headers and data for the API request
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    data = {
        'model': 'tts-1',
        'voice': 'alloy',
        'input': text
    }

    # Make the POST request to OpenAI's endpoint
    url = 'https://api.openai.com/v1/audio/speech'
    response = requests.post(url, json=data, headers=headers)

    # Check if the request was successful and handle the response
    if response.status_code == 200:
        # Return the audio data
        return response.content
    else:
        st.error(f"Failed to create speech. Status code: {response.status_code} Response: {response.text}")
        return None

# Main interface
mode = st.sidebar.radio("Choose a mode:", ["AI Mentor ChatBot", "AI Image Generator", "Text-to-Speech", "Audio Transcription"])

# Create a text input for user
user_input = st.text_input('Enter your question:', key='user_input', value=st.session_state['user_input'])

# Create layout for submit and clear buttons in a single row
col1, col2 = st.columns([2, 1])

# Create a submit button
with col1:
        if mode != "Text-to-Speech":
            if st.button("Submit") and user_input:
                submit(user_input)
                # Check the mode and process accordingly
                if mode == "AI Mentor ChatBot" and st.session_state.entered_prompt:
                    with st.spinner('Generating response...'):
                        # Get user query
                        user_query = st.session_state.entered_prompt

                        # Append user query to past queries
                        st.session_state.past.append(user_query)

                        # Generate response
                        output = generate_response()

                        # Append AI response to generated responses
                        st.session_state.generated.append(output)

                    # Display the chat history
                    if st.session_state['generated']:
                        for i in reversed(range(len(st.session_state['generated']))):
                            # Display user message
                            user_msg = st.session_state['past'][i]
                            display_message(user_msg, is_user=True)
                            
                            # Display AI response
                            bot_msg = st.session_state["generated"][i]
                            display_message(bot_msg, is_user=False)

                elif mode == "AI Image Generator" and user_input:
                    with st.spinner('Generating image...'):
                        generate_image(user_input)

# Text-to-Speech Interface
if mode == "Text-to-Speech":
    st.subheader("Text-to-Speech Conversion")
    tts_input = st.text_area("Enter text to convert into speech:", key="tts_input")
    if st.button("Convert to Speech"):
        if tts_input:  # Check if input is not empty
            with st.spinner('Generating speech...'):
                speech_data = convert_text_to_speech(tts_input)
                if speech_data:
                        # Save the audio file
                        speech_file_path = Path('TTS_output.mp3')
                        with open(speech_file_path, 'wb') as audio_file:
                            audio_file.write(speech_data)

                        # Read the saved audio file's binary data
                        with open(speech_file_path, 'rb') as audio_file:
                            audio_data = audio_file.read()

                        # Display the audio player with the binary data
                        st.audio(audio_data, format='audio/mp3')
                        st.success("Speech generation completed!")

# Audio Transcription Interface
if mode == "Audio Transcription":
    st.subheader("Audio Transcription")
    uploaded_file = st.file_uploader("Upload your audio file (.mp3):", type=['mp3'])

    # Ensure transcription_text is in session_state
    if 'transcription_text' not in st.session_state:
        st.session_state['transcription_text'] = ''

    # Language selection logic
    language_option = st.selectbox("Select the language for transcription:", ["English", "Urdu"])
    language_code = 'en' if language_option == "English" else 'ur'

    if st.button("Transcribe Audio"):
        if uploaded_file is not None:
            with st.spinner('Transcribing audio...'):
                # Transcribe the audio file
                transcription = transcribe_audio(uploaded_file, language_code)
                if transcription:
                    st.session_state['transcription_text'] = transcription
                    st.text_area("Transcription:", transcription, height=150)
                else:
                    st.error("Failed to transcribe the audio.")

    # Button to export the transcription to a DOCX file
    if st.session_state['transcription_text']:
        if st.button("Export to DOCX"):
            doc_path = save_transcription_to_docx(st.session_state['transcription_text'])
            if doc_path:
                st.success(f"Transcription exported to {doc_path}.")
                # Provide a download link
                with open(doc_path, "rb") as file:
                    st.download_button(
                        label="Download Transcription",
                        data=file,
                        file_name=os.path.basename(doc_path),
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
            else:
                st.error("Failed to export the transcription.")

with col2:
    if st.button("Clear 🗑️", on_click=clear_text):
        pass  # Clearing is handled by the callback function