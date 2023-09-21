from langchain import HuggingFaceHub, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import find_dotenv, load_dotenv 
import requests
import os  
from flask import Flask, render_template, request
import chainlit as cl


huggingfacehub_api_token = os.environ['HUGGINGFACEHUB_API_TOKEN']
huggingfacehub_api_token = "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"
load_dotenv(find_dotenv())

repo_id = "microsoft/phi-1_5"
lm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token, 
                     repo_id=repo_id, 
                     model_kwargs={"temperature":0.7, "max_new_tokens":2000})

template = """

    You are an executive counsellor and an empathetic support figure. In your role, you have the unique responsibility of guiding high-profile executives through the challenges they face in both their personal and professional lives, including family and relationship issues, with warmth, compassion, and professionalism.
    {history}
    Me:{human_input}
    Counsellor:
    """


    
def factory(human_input):
    prompt = PromptTemplate(
        input_variables=["history", "human_input"],
        template=template
    )
    llm_chain = LLMChain(
        llm=HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token, repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.2}),
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(ai_prefix="Executive Counsellor")
    )
    output = llm_chain.predict(human_input=human_input)

    return output


#web GUI
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/send_message', methods=['POST'])
def send_message():
    human_input=request.form['human_input']
    message = factory(human_input)
    return message or ''

if __name__ == '__main__':
    app.run(debug=True)



