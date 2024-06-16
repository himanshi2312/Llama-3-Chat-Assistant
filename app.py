from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import logging
import streamlit as st
import os
from dotenv import load_dotenv


load_dotenv()

# environment variables for LangChain
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Initialize the Llama-3 model
def intialize_llama3():
    try:
        new_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant. Please respond to user queries"),
                ("user", "Question: {question}"),
            ]
        )

        llm = Ollama(model="llama3")  

        format_output = StrOutputParser()

        chatbot_pipeline = new_prompt | llm | format_output

        return chatbot_pipeline

    except Exception as e:
        logging.error(f"Failed to initialize llama3: {e}")
        raise


chatbot_pipeline = intialize_llama3()

def main():
    
    st.title('Infinite loop with Llama-3')

    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    
    with st.form(key='chat_form'):
        input_text = st.text_input("Search the topic you want")
        submit_button = st.form_submit_button(label="Send Message")

    if input_text and submit_button:
        try:
            response = chatbot_pipeline.invoke({'question': input_text})
            st.session_state.conversation.append(input_text)
            st.session_state.conversation.append(response)
        except Exception as e:
            logging.error(f"Error to invoke llama3: {e}")
            st.session_state.conversation.append("Sorry, an error occurred while processing your request.")

    for message in st.session_state.conversation:
        st.write(message)
    
    

if __name__ == "__main__":
    main()
