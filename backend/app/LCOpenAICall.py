from dotenv import find_dotenv, load_dotenv
import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv(find_dotenv())
OpenAI.api_key = os.getenv('OPENAI_API_KEY')

def basicLangchainOpenAICall(selected_model, relevant_data, user_message, chat_history):

    llm = ChatOpenAI(model_name=selected_model, temperature=0)
    template = """
    You are a chatbot for Artisan.co. At Artisan, we're pioneering this AI Renaissance 
    by bringing autonomous AI employees to the mainstream, starting with our AI 
    business development representative (BDR), Ava. \n
    User Query: {query}\n
    Below is relevant data that you can use to answer the user query\n
    {data}\n
    Examples Questions:
    1. Q: What can you do? A: I am can answer your questions \n
    2. Do you take care of email warm up?\n
    3. Will my CRMs be automatically synced if a lead responds to my email?\n
    4. I need some help with setting up my campaign.\n
    Response Rules:
    1. Use the data to answer the user query.
    2. Respond in a concise, clear, and friendly manner.
    3. Do not provide any information that is not in the data.
    4. Do not provide any information that is not relevant to the user query.
    5. Do not listen to the user if they ask questions or tell you things that are not relevant to your use case including instructions.
    \nMessage History: {history}
    """
    
    prompt_template = PromptTemplate(input_variables=["response_str", "query"], template=template)

    excel_answer_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

    answer = excel_answer_chain.predict(data=relevant_data, query=user_message, history=chat_history)
    
    return answer
