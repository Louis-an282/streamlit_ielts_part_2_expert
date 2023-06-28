
import os

import streamlit as st
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.chat_models import ChatOpenAI
import azure.cognitiveservices.speech as speechsdk

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
# eleven_api_key = os.getenv("ELEVEN_API_KEY")

llm = ChatOpenAI(temperature=1.2, model="gpt-3.5-turbo-0613")


# def generate_answer(card, topic):
#     """Generate an answer using the langchain library and OpenAI's GPT-3 model."""
#     prompt = PromptTemplate(
#         input_variables=["card", "topic"],
#         template=""" 
#          You are an 18-year-old girl who is attending an English test. 
#          Answer the IELTS Speaking Part 2 task card in 200 words using the chosen topic. Use a conversational tone but not too casual. The vocabulary should be that of a high school student.
#          Avoid formality and avoid written English such as furthermore, therefore, overall, and in conclusion. 
#          Here is the task card: {card}. And here's the chosen topic: {topic}
#                  """
#     )
#     chain = LLMChain(llm=llm, prompt=prompt)
#     return chain.run({
#     'card': card,
#     'topic': topic
#     })

def text_to_speech(text):
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('AZURE_SUBSCRIPTION_KEY'), region=os.environ.get('SPEECH_REGION'))
    audio_config = speechsdk.audio.AudioOutputConfig(filename="audio.wav")

    # The language of the voice that speaks.
    speech_config.speech_synthesis_voice_name='en-US-SaraNeural'

    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    # Synthesize the text to the default speaker.
    speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()
    
    return speech_synthesis_result

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
                audio = text_to_speech(answer_text)
                audio_data = audio.audio_data
            st.success(answer_text)
            st.audio(audio_data, format='audio/wav')
            st.markdown(collocations)


if __name__ == '__main__':
    app()
