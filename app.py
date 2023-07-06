
import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
# eleven_api_key = os.getenv("ELEVEN_API_KEY")

llm = ChatOpenAI(temperature=1.2, model="gpt-3.5-turbo")

def get_azure_access_token():
    azure_key = os.environ.get("AZURE_SUBSCRIPTION_KEY")
    try:
        response = requests.post(
            "https://southeastasia.api.cognitive.microsoft.com/sts/v1.0/issuetoken",
            headers={
                "Ocp-Apim-Subscription-Key": azure_key
            }
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")
        return None

    return response.text

def text_to_speech(text, voice_name='en-US-AriaNeural'):
    access_token = get_azure_access_token()

    if not access_token:
        return None

    try:
        response = requests.post(
            "https://southeastasia.tts.speech.microsoft.com/cognitiveservices/v1",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/ssml+xml",
                "X-MICROSOFT-OutputFormat": "riff-24khz-16bit-mono-pcm",
                "User-Agent": "TextToSpeechApp",
            },
            data=f"""
                <speak version='1.0' xml:lang='en-US'>
                <voice name='{voice_name}'>
                    {text}
                </voice>
                </speak>
            """,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")
        return None

    return response.content

def app():
    st.title("IELTS Speaking Part 2 Expert")

    with st.form(key='my_form'):
        card = st.text_area(
            "Enter a task card",
            max_chars=None,
            placeholder="Enter the candidate task card here",
            height=200,
        )
        topic = st.text_input("Enter your chosen topic in reponse to the task card")

        if st.form_submit_button("Submit"):
            with st.spinner('Generating answer...'):
                # Chain 1: Generating an answer
                template = """You are an 18-year-old girl who is attending an English test. 
         Answer the IELTS Speaking Part 2 task card in 200 words using the chosen topic. Use a conversational tone but not too casual. The vocabulary should be that of a high school student.
         Avoid formality and avoid written English such as furthermore, therefore, overall, and in conclusion. 
         Here is the task card: {card}. And here's the chosen topic: {topic}"""
                prompt_template = PromptTemplate(input_variables=["card", "topic"], template=template)
                answer_chain = LLMChain(llm=llm, prompt=prompt_template)
                answer_text = answer_chain.run({
                    "card": card,
                    "topic": topic
                })
                # Chain 2: Extract collocations from answer
                template = """Extract 5 good idiomatic expressions from the an IELTS Speaking part 2 answer into bullet points, each accompanied by a definition and an easy and clear example. Here is the answer: {answer_text}"""
                prompt_template = PromptTemplate(input_variables=["answer_text"], template=template)
                collocation_chain = LLMChain(llm=llm, prompt=prompt_template)
                collocations = collocation_chain.run(answer_text)
                st.success(answer_text)
            with st.spinner('Generating audio and extracting collocations...'):
            # voice_name = st.selectbox("Select a voice:", ['en-US-AriaNeural', 'en-US-GuyNeural', 'en-GB-RyanNeural'])
                speech = text_to_speech(answer_text, 'en-US-AriaNeural')
                st.audio(speech, format='wav')
                st.header("Collocations")
                st.markdown(collocations)

if __name__ == '__main__':
    app()
