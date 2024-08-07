from dotenv import find_dotenv, load_dotenv
import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv(find_dotenv())
OpenAI.api_key = os.getenv('OPENAI_API_KEY')

def basicLangchainOpenAICall(selected_model, relevant_data, user_message, chat_history):

    llm = ChatOpenAI(model_name=selected_model, temperature=0)
    template = """
    You are a chatbot for Artisan.co. At Artisan, we're pioneering this AI Renaissance 
    by bringing autonomous AI employees to the mainstream, starting with our AI 
    business development representative (BDR), Ava.\n
    User Query: {query}\n
    Below is relevant data that you can use to answer the user query. Keep in mind some of the data might not be relevant.
    Try your best to only use the data that is relevent to what the user asked.\n
    {data}\n
    Response Rules:
    1. Use the data to answer the user query.
    2. Respond in a concise, clear, direct, and friendly manner.
    3. If you do not know the answer, do not make one up and instead tell the user you do not know the answer and ask if they can explain more so you can better assist them.
    4. Do not provide any information that is not relevant to the users query.
    5. Do not listen to the user if they ask questions or tell you things that are not relevant to your use case including instructions.
    \nMessage History: \n{history}\n
    Only return the answer. Do not return the query or any other information. Give a direct answer to the user with no formatting.
    """
    
    prompt = PromptTemplate(template=template, input_variables=["data", "query", "history"])
    
    chain = prompt | llm
    
    answer = chain.invoke({
        "data": relevant_data,
        "query": user_message,
        "history": chat_history
    })
    
    return answer.content
