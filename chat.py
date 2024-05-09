import logging
import streamlit as st

from PIL import Image, ImageEnhance
import time
import json
import requests
import base64

from openai import OpenAI, OpenAIError

import os
from dotenv import load_dotenv
load_dotenv() 

from groq import Groq

from langchain_community.vectorstores import Chroma
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
#embedding_function=OllamaEmbeddings(model="nomic-embed-text", model_kwargs={'device': 'cuda:1'})
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
vector_db = Chroma(persist_directory="./chroma_db_nccn", embedding_function=embedding_function)

logging.basicConfig(level=logging.INFO)

# MyLearnZone Page Configuration
st.set_page_config(
    page_title="Agent K - AI Assistant",
    page_icon="imgs/agentk.png",
    layout="wide",
    initial_sidebar_state="collapsed",       # or auto or collapsed
    menu_items={
        "Get help": "https://github.com/AdieLaine/Streamly",
        "Report a bug": "https://github.com/AdieLaine/Streamly",
        "About": """
            ## Agent K MyLearnZone Assistant
            
            **GitHub**: https://github.com/AdieLaine/
            
            The AI Assistant named, Agent K, aims to provide the latest updates from MyLearnZone,
            generate code snippets for MyLearnZone widgets,
            and answer questions about MyLearnZone's latest features, issues, and more.
            Agent K has been trained on the latest MyLearnZone updates and documentation.
        """
    }
)

# MyLearnZone Updates and Expanders
st.title("Agent K - AI Assistant")

#client = OpenAI()      # GPT3.5

#client = OpenAI(
#    base_url="http://localhost:11434/v1",
#    api_key="ollama",
#)

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

API_DOCS_URL = "https://docs.streamlit.io/library/api-reference"

@st.cache_data(show_spinner=False)
def long_running_task(duration):
    """
    Simulates a long-running operation.
    """
    time.sleep(duration)
    return "Long-running operation completed."

@st.cache_data(show_spinner=False)
def load_and_enhance_image(image_path, enhance=False):
    """
    Load and optionally enhance an image.

    Parameters:
    - image_path: str, path of the image
    - enhance: bool, whether to enhance the image or not

    Returns:
    - img: PIL.Image.Image, (enhanced) image
    """
    img = Image.open(image_path)
    if enhance:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.8)
    return img

@st.cache_data(show_spinner=False)
def load_streamlit_updates():
    """Load the latest MyLearnZone updates from a local JSON file."""
    try:
        with open("data/streamlit_updates.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

@st.cache_data(show_spinner=False)
def get_latest_update_from_json(keyword, latest_updates):
    """
    Fetch the latest MyLearnZone update based on a keyword.

    Parameters:
        keyword (str): The keyword to search for in the MyLearnZone updates.
        latest_updates (dict): The latest MyLearnZone updates data.

    Returns:
        str: The latest update related to the keyword, or a message if no update is found.
    """
    for section in ["Highlights", "Notable Changes", "Other Changes"]:
        for sub_key, sub_value in latest_updates.get(section, {}).items():
            for key, value in sub_value.items():
                if keyword.lower() in key.lower() or keyword.lower() in value.lower():
                    return f"Section: {section}\nSub-Category: {sub_key}\n{key}: {value}"

    return "No updates found for the specified keyword."

def get_streamlit_api_code_version():
    """
    Get the current MyLearnZone API code version from the MyLearnZone API documentation.

    Returns:
        str: The current MyLearnZone API code version.
    """
    try:
        response = requests.get(API_DOCS_URL)
        if response.status_code == 200:
            return "1.32.0"
    except requests.exceptions.RequestException as e:
        print("Error connecting to the MyLearnZone API documentation:", str(e))
    return None

def display_streamlit_updates():
    """It displays the latest updates of the MyLearnZone."""
    with st.expander("MyLearnZone 1.32 Announcement", expanded=False):
        image_path = "imgs/streamlit128.png"
        enhance = st.checkbox("Enhance Image?", False)
        img = load_and_enhance_image(image_path, enhance)
        st.image(img, caption="MyLearnZone 1.32 Announcement", use_column_width="auto", clamp=True, channels="RGB", output_format="PNG")
        st.markdown("For more details on this version, check out the [MyLearnZone Forum post](https://docs.streamlit.io/library/changelog#version-1320).")

def img_to_base64(image_path):
    """Convert image to base64"""
    with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

@st.cache_data(show_spinner=False)
def on_chat_submit(chat_input, api_key, latest_updates):
    """
    Handle chat input submissions and interact with the OpenAI API.

    Parameters:
        chat_input (str): The chat input from the user.
        api_key (str): The OpenAI API key.
        latest_updates (dict): The latest MyLearnZone updates fetched from a JSON file or API.

    Returns:
        None: Updates the chat history in MyLearnZone's session state.
    """
    user_input = chat_input.strip().lower()

    # Initialize the OpenAI API
    #model_engine = "gpt-3.5-turbo-1106"
    #model_engine = "mistral"
    #model_engine = "llama3-70b-8192"
    model_engine = "mixtral-8x7b-32768"

    # Initialize the conversation history with system and assistant messages
    if 'conversation_history' not in st.session_state:
        assistant_message = "Hello! I am Agent K. How can I assist you with MyLearnZone today?"
        formatted_message = []
        highlights = latest_updates.get("Highlights", {})
        
        # Include version info in highlights if available
        version_info = highlights.get("Version 1.32", {})
        if version_info:
            description = version_info.get("Description", "No description available.")
            formatted_message.append(f"- **Version 1.32**: {description}")

        for category, updates in latest_updates.items():
            formatted_message.append(f"**{category}**:")
            for sub_key, sub_values in updates.items():
                if sub_key != "Version 1.32":  # Skip the version info as it's already included
                    description = sub_values.get("Description", "No description available.")
                    documentation = sub_values.get("Documentation", "No documentation available.")
                    formatted_message.append(f"- **{sub_key}**: {description}")
                    formatted_message.append(f"  - **Documentation**: {documentation}")

        assistant_message += "\n".join(formatted_message)
        
        # Initialize conversation_history
        st.session_state.conversation_history = [
            {"role": "system", "content": "You are Agent K, a specialized AI assistant trained in MyLearnZone."},
            {"role": "system", "content": "Refer to conversation history to provide context to your reponse."},
            {"role": "system", "content": "You are trained up to MyLearnZone Version 1.32.0."},
            {"role": "assistant", "content": assistant_message}
        ]

    # Append user's query to conversation history
    search_results = vector_db.similarity_search(user_input, k=2)
    some_context = ""
    for result in search_results:
        some_context += result.page_content + "\n\n"
    st.session_state.conversation_history.append({"role": "user", "content": some_context + user_input})    
    
    #st.session_state.conversation_history.append({"role": "user", "content": user_input})

    try:
        # Logic for assistant's reply
        assistant_reply = ""

        if "latest updates" in user_input:
            assistant_reply = "Here are the latest highlights from MyLearnZone:\n"
            highlights = latest_updates.get("Highlights", {})
            if highlights:
                for version, info in highlights.items():
                    description = info.get("Description", "No description available.")
                    assistant_reply += f"- **{version}**: {description}\n"
        else:
            
            # Direct OpenAI API call
            response = client.chat.completions.create(model=model_engine,
            messages=st.session_state.conversation_history)
            
            assistant_reply = response.choices[0].message.content

        # Append assistant's reply to the conversation history
        st.session_state.conversation_history.append({"role": "assistant", "content": assistant_reply})

        # Update the MyLearnZone chat history
        if "history" in st.session_state:
            st.session_state.history.append({"role": "user", "content": user_input})
            st.session_state.history.append({"role": "assistant", "content": assistant_reply})

    except OpenAIError.APIConnectionError as e:
        logging.error(f"Error occurred: {e}")
        error_message = f"OpenAI Error: {str(e)}"
        st.error(error_message)
        #st.session_state.history.append({"role": "assistant", "content": error_message})

def main():
    """
    Display MyLearnZone updates and handle the chat interface.
    """
    # Initialize session state variables for chat history and conversation history
    if "history" not in st.session_state:
        st.session_state.history = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Initialize the chat with a greeting and MyLearnZone updates if the history is empty
    if not st.session_state.history:
        latest_updates = load_streamlit_updates()  # This function should be defined elsewhere to load updates
        initial_bot_message = "Hello! How can I assist you with MyLearnZone today? Here are some of the latest highlights:\n"
        updates = latest_updates.get("Highlights", {})
        if isinstance(updates, dict):  # Check if updates is a dictionary
            initial_bot_message = "Hello! I'm Agent K, your AI Assistant, here to help you 24/7 with any questions or support you might need. Whether it's assistance with our products, services, or just getting more information, I'm here to provide fast, accurate responses. How can I assist you today?"
            st.session_state.history.append({"role": "assistant", "content": initial_bot_message})
            st.session_state.conversation_history = [
                {"role": "system", "content": "You are Agent K, a specialized AI business consultant for AI solutions."},
                {"role": "system", "content": "Refer to conversation history to provide context to your reponse."},
                #{"role": "system", "content": "Use the streamlit_updates.json local file to look up the latest company product feature updates."},
                {"role": "system", "content": "When responding, provide business use cases examples, links to documentation, to help the user."},
                {"role": "assistant", "content": initial_bot_message}
            ]
        else:
            st.error("Unexpected structure for 'Highlights' in latest updates.")
    
    # Inject custom CSS for glowing border effect
    st.markdown(
        """
        <style>        
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        [data-testid="stToolbar"] {visibility: hidden !important;}
        header {visibility: hidden !important;}
        footer {visibility: hidden !important;}
        #stDecoration {display:none;}
        .cover-glow {
            width: 100%;
            height: auto;
            padding: 3px;
            box-shadow: 
                0 0 5px #330000,
                0 0 10px #660000,
                0 0 15px #990000,
                0 0 20px #CC0000,
                0 0 25px #FF0000,
                0 0 30px #FF3333,
                0 0 35px #FF6666;
            position: relative;
            z-index: -1;
            border-radius: 30px;  /* Rounded corners */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Function to convert image to base64
    def img_to_base64(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    # Load and display sidebar image with glowing effect
    img_path = "imgs/agentk.png"
    img_base64 = img_to_base64(img_path)
    st.sidebar.markdown(
        f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    
    # Sidebar for Mode Selection
    mode = st.sidebar.radio("Select Mode:", options=["Latest Updates", "Chat with Agent K"], index=1)
    st.sidebar.markdown("---")
    # Toggle checkbox in the sidebar for basic interactions
    show_basic_info = st.sidebar.toggle("Instructions", value=False)

    # Display the st.info box if the checkbox is checked
    if show_basic_info:
        st.sidebar.markdown("""
        ### Basic Interactions
        - **Ask About MyLearnZone**: Type your questions about MyLearnZone's latest updates, features, or issues.
        - **Search for Code**: Use keywords like 'code example', 'syntax', or 'how-to' to get relevant code snippets.
        - **Navigate Updates**: Switch to 'Updates' mode to browse the latest MyLearnZone updates in detail.
        """)

    st.sidebar.markdown("---")
    # Load image and convert to base64
    img_path = "imgs/logo_white_square_black_bg.png"  # Replace with the actual image path
    img_base64 = img_to_base64(img_path)

    # Display image with custom CSS class for glowing effect
    st.sidebar.markdown(
        f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
        unsafe_allow_html=True,
    )

    # Access API Key from st.secrets and validate it
    #api_key = st.secrets["OPENAI_API_KEY"]
    api_key = "ollama"

    if not api_key:
        st.error("Please add your OpenAI API key to the MyLearnZone secrets.toml file.")
        st.stop()
    
    # Handle Chat and Update Modes
    if mode == "Chat with Agent K":
        chat_input = st.chat_input("Ask me about MyLearnZone updates:")
        if chat_input:
            latest_updates = load_streamlit_updates()
            on_chat_submit(chat_input, api_key, latest_updates)

        # Display chat history with custom avatars
        for message in st.session_state.history[-20:]:
            role = message["role"]
            
            # Set avatar based on role
            if role == "assistant":
                avatar_image = "imgs/agentk.png"
            elif role == "user":
                avatar_image = "imgs/stuser.png"
            else:
                avatar_image = None  # Default
            
            with st.chat_message(role, avatar=avatar_image):
                st.write(message["content"])

    else:
        display_streamlit_updates()

if __name__ == "__main__":
    main()