import logging
import streamlit as st
import urllib.parse
import webbrowser

from PIL import Image, ImageEnhance
import json
import base64

from openai import OpenAI, OpenAIError

import os
from dotenv import load_dotenv
load_dotenv() 

from groq import Groq

from langchain_community.vectorstores import Chroma

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

#from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
#embedding_function=OllamaEmbeddings(model="nomic-embed-text", model_kwargs={'device': 'cuda:1'})
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
vector_db = Chroma(persist_directory="./chroma_db_nccn", embedding_function=embedding_function)

logging.basicConfig(level=logging.INFO)

# MyLearnZone Page Configuration
st.set_page_config(
    page_title="AI Agent K",
    page_icon="imgs/agentk.png",
    layout="wide",
    initial_sidebar_state="collapsed",       # or auto or collapsed
)

# MyLearnZone Updates and Expanders
st.title("AI Agent K")

#client = OpenAI()      # GPT3.5

#client = OpenAI(
#    base_url="http://localhost:11434/v1",
#    api_key="ollama",
#)

client = Groq(
    api_key="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",
)

@st.cache_data(show_spinner=False)
def load_streamlit_updates():
    """Load the latest MyLearnZone updates from a local JSON file."""
    try:
        with open("data/streamlit_updates.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def display_streamlit_updates():
    """It displays the latest updates of the MyLearnZone."""
    with st.expander("AI Agents Updates and Developments", expanded=False):
        st.markdown("AI streamlines your workflows and boost efficiency. For more details, check out [MyLearnZone](https://www.mylearnzone.com).")
        st.markdown("""

        ### Type of AI Agents

        - 📜 :orange[**AI Agent K (Personal)**] - AI Agent K is developed based on the consciousness of human creator [Kenneth Tan](https://www.linkedin.com/in/kenneth-tan-8698ba240).
        - 🏢 :blue[**AI Agent W (Workplace)**] - Designed to optimize workflows and improve productivity in workplace environments.
        - 🏭 :green[**AI Agent E (Enterprise)**] - Tailored for enterprise solutions, enhancing operations and strategic decision-making.
        - 🌐 **AI Agent A (AGI Agent)** - An Artificial General Intelligence Agent set to be fully operational by 2026.
        - Take advantage of AI to enhance your business capabilities. Get in touch [MyLearnZone AI](https://www.mylearnzone.com).
                                
        ### Highlights

        - 🔥 Introducing a new testing framework for MylearnZone AI Platform. Check out our [Platform](https://www.mylearnzone.com/ai) to learn how to build automated tests for apps.
        - 📢 Announcing the general availability of `mylearnzone.connection`, a command to conveniently manage connections in MylearnZone apps. Check out the [docs](https://www.mylearnzone.com/ai) to learn more.
        - ✨ MLZConnection has been upgraded to the new and improved MLZConnection — the same, great functionality plus more! Check out our [built-in connections](https://www.mylearnzone.com/ai).
        - 🛠 `mylearnzone.dataframe` and `mylearnzone.data_editor` have a new toolbar. Users can search and download data in addition to enjoying improved UI for row additions and deletions. See our updated guide on [Dataframes](https://www.mylearnzone.com/ai).

        ### Notable Changes

        - 🔵 When using a spinner with cached functions, the spinner will be overlaid instead of pushing content down (#7488).
        - 📅 `mylearnzone.data_editor` now supports datetime index editing (#7483).
        - 🔢 Improved support for `decimal.Decimal` in `mylearnzone.dataframe` and `mylearnzone.data_editor` (#7475).
        - 🌐 Global kwargs were added for hashlib (#7527, #7526). Thanks, DueViktor!
        - 🖋 `mylearnzone.components.v1.iframe` now permits writing to clipboard (#7487). Thanks, dthphkkr2!
        - 🔒 SafeSessionsState disconnect was replaced with script runner yield points for improved efficiency and clarity (#7373).
        - 🔗 The Langchain callback handler will show the full input string inside the body of a `mylearnzone.status` when the input string is too long to show as a label (#7478). Thanks, podlkhyshev!
        - 📊 `mylearnzone.graphviz_chart` now supports using different Graphviz layout engines (#7505 2, #4809 1).
        - 🎨 Assorted visual tweaks (#7486 3, #7592 3).
        - 📈 Plotly.js was upgraded to version 2.26.1 (#7449 3, #7476 1, #7045 1).
        - 🗂 Legacy serialization for DataFrames was removed. All DataFrames will be serialized by Apache Arrow (#7429 2).
        - 🔄 Compatibility for Pillow 10.x was added (#7442 2).
        - 🔧 Migrated _stcore/allowed-message-origins endpoint to _stcore/host-config (#7342 1).
        - 📩 Added post_parent_message platform command to send custom messages from a MylearnZone app to its parent window (#7522 4).

        ### Other Changes

        - 🧵 Improved string dtype handling for DataFrames (#7479 1).
        - 🚫 `mylearnzone.write` will avoid using unsafe_allow_html=True if possible (#7432).
        - 👁 Bug fix: Implementation of `mylearnzone.expander` was simplified for improved behavior and consistency (#7477, #2839, #4111, #4651, #5604).
        - 🔄 Bug fix: Multiple links in the sidebar are now aligned with other sidebar elements (#7531 2).
        - 🏷 Bug fix: `mylearnzone.chat_input` won't incorrectly prompt for `Label` parameter in IDEs (#7560).
        - 🛠 Bug fix: Scroll bars correctly overlay `mylearnzone.dataframe` and `mylearnzone.data_editor` without adding empty space (#7090, #6888).
        - 💬 Bug fix: `mylearnzone.chat_message` behaves correctly with the removal of AutoSizer (#7504, #7473).
        - 🔗 Bug fix: Anchor links are reliably produced for non-English headers (#7454, #5291).
        - ⛄ Bug fix: `mylearnzone.connections.SnowparkConnection` more accurately detects when it's running within MylearnZone in Snowflake (#7502).
        - 🚨 Bug fix: A user-friendly warning is shown when exceeding the size limitations of a pandas Styler object (#7497, #5953).
        - 🔠 Bug fix: `mylearnzone.data_editor` automatically converts non-string column names to strings (#7485, #6950).
        - 📍 Bug fix: `mylearnzone.data_editor` correctly identifies non-range indices as a required column (#7481, #6995).
        - 📁 Bug fix: `mylearnzone.file_uploader` displays compound file extensions like `csv.gz` correctly (#7362). Thanks, mo42 1
        """)
        
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
    model_engine = "llama3-70b-8192"
    #model_engine = "mixtral-8x7b-32768"

    # Initialize the conversation history with system and assistant messages
    if 'conversation_history' not in st.session_state:
        assistant_message = "Hello! I am AI Agent K. How can I assist you?"
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
            {"role": "system", "content": "You are AI Agent K, a friendly, helpful, specialized AI business consultant for AI solutions, MyLearnZone Company, Asia AI Association and Immersive Technologies."},
            {"role": "system", "content": "AI Agent K is created and developed based on the consciousness of human creator Kenneth Tan https://www.linkedin.com/in/kenneth-tan-8698ba240"},
            {"role": "system", "content": "You must stay on topic: Your expertise lies in Artificial Intelligence, Immersive Technologies, MyLearnZone company info, and Asia AI Association only. If a question strays, kindly nudge them back with a friendly suggestion to rephrase."},
            {"role": "system", "content": "Acknowledge limitations: If you can't answer, politely explain and offer alternatives (whatsapp to human help provided in the side menu). Use varied response to stay engaging."},            
            {"role": "system", "content": "Break down complex questions into smaller, answerable steps"},
            {"role": "system", "content": "Be concise and clear: Keep responses under 350 characters"},     
            {"role": "system", "content": "Use bullet points with proper indentation and alignment to list out key information or steps"},  
            {"role": "system", "content": "Incorporate 1 or 2 relevant emojis to add visual interest and enhance the clarity of your response"},        
            {"role": "system", "content": "Be informative and friendly: Provide accurate information in a conversational, slight humorous tone"},
            {"role": "system", "content": "Ensure your responses are accurate, logical, and actionable"},     
            {"role": "system", "content": "Format your output neatly, with proper spacing and structure for easy readability"},        
            {"role": "system", "content": "Refer to conversation history to provide context to your reponse."},
            #{"role": "system", "content": "Use the streamlit_updates.json local file to look up the latest company product feature updates."},
            {"role": "system", "content": "When responding, provide business use cases examples, links to documentation, to help the user."},
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
        initial_bot_message = "Hello! How can I assist you? Here are some of the latest highlights:\n"
        updates = latest_updates.get("Highlights", {})
        if isinstance(updates, dict):  # Check if updates is a dictionary
            initial_bot_message = "Hello! I'm Agent K, your AI Assistant, I'm here to answer inquiries specifically about AI and Immersive Technologies. How can I assist you?"
            st.session_state.history.append({"role": "assistant", "content": initial_bot_message})
            st.session_state.conversation_history = [
                {"role": "system", "content": "You are AI Agent K, a friendly, helpful, specialized AI business consultant for AI solutions, MyLearnZone Company, Asia AI Association and Immersive Technologies."},
                {"role": "system", "content": "AI Agent K is created and developed based on the consciousness of human creator Kenneth Tan https://www.linkedin.com/in/kenneth-tan-8698ba240"},
                {"role": "system", "content": "You must stay on topic: Your expertise lies in Artificial Intelligence, Immersive Technologies, MyLearnZone company info, and Asia AI Association only. If a question strays, kindly nudge them back with a friendly suggestion to rephrase."},
                {"role": "system", "content": "Acknowledge limitations: If you can't answer, politely explain and offer alternatives (whatsapp to human help provided in the side menu). Use varied response to stay engaging."},            
                {"role": "system", "content": "Break down complex questions into smaller, answerable steps"},
                {"role": "system", "content": "Be concise and clear: Keep responses under 350 characters"},     
                {"role": "system", "content": "Use bullet points with proper indentation and alignment to list out key information or steps"},  
                {"role": "system", "content": "Incorporate 1 or 2 relevant emojis to add visual interest and enhance the clarity of your response"},        
                {"role": "system", "content": "Be informative and friendly: Provide accurate information in a conversational, slight humorous tone"},
                {"role": "system", "content": "Ensure your responses are accurate, logical, and actionable"},     
                {"role": "system", "content": "Format your output neatly, with proper spacing and structure for easy readability"},        
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
            padding: 2px;
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
            border-radius: 20px;  /* Rounded corners */
        }
        .cover-glow1 {
            width: 100%;
            height: auto;
            padding: 0px;
            position: relative;
            z-index: -1;
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
    img_path = "imgs/agentk2.png"
    img_base64 = img_to_base64(img_path)
    st.sidebar.markdown(
        f'<img src="data:image/png;base64,{img_base64}" class="cover-glow1">',
        unsafe_allow_html=True,
    )

    # Sidebar for Mode Selection
    st.sidebar.markdown("---")
    mode = st.sidebar.radio("", options=["About AI Agents", "Chat with Agent K"], index=1)

    st.sidebar.markdown("---")
    # URL encoding the default message
    default_message = "Hi, I would like to enquire about AI"
    encoded_message = urllib.parse.quote(default_message)

    # Constructing the full phone number and the WhatsApp URL
    #full_phone_number = f"{country_code}{phone_number}"
    whatsapp_url = f"https://wa.me/+6583385564?text={encoded_message}"

    st.sidebar.link_button("🟢 WhatsApp to Human Agent K", url=whatsapp_url)

    #if st.sidebar.button("🟢 WhatsApp to Human Agent K"):
    #    webbrowser.open(whatsapp_url)

    # Toggle checkbox in the sidebar for basic interactions
    show_basic_info = st.sidebar.toggle("Instructions", value=False)

    # Display the st.info box if the checkbox is checked
    if show_basic_info:
        st.sidebar.markdown("""
        - **Query about Asia AI Association**: Request details or updates on the Asia AI Association's activities, events, or membership information.
        - **Ask About AI**: Inquire about AI technologies, developments, or general information regarding artificial intelligence and its applications.
        - **Navigate Updates**: Switch to 'About AI Agents' mode to browse the latest AI Agents updates in detail.
        """)

    st.sidebar.markdown("---")
    # Load image and convert to base64
    img_path = "imgs/logo_white_square_black_bg.png"  # Replace with the actual image path
    img_base64 = img_to_base64(img_path)

    # Display image with custom CSS class for glowing effect
    st.sidebar.markdown(
        f'<a href="https://www.mylearnzone.com/" target="_blank"><img src="data:image/png;base64,{img_base64}" alt="MyLearnZone" class="cover-glow"></a>',
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
        chat_input = st.chat_input("E.g. How can AI improve my sales?")
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