import os
import time
import pandas as pd
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, HuggingFacePipeline 
from sklearn.metrics import accuracy_score
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from transformers import pipeline, LlamaTokenizer, AutoModelForCausalLM


load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


medi_template = '''You are a medical advisor and an expert in medical science. You will be given a medical related sentence and a json with 4 options to complete the sentence. You have to determine which sentence from the 4 options is more likely to come after the ctx, then you've to reply the correct option that completes the sentence given in ctx.
When completing the sentence, you have to remember some criterias: 1. Follow grammatical rules of completing sentences. 2. Do medical analysis to find out which sentence is more likely to come.

Reply with only 0/1/2 or 3. The numbers denote the option given in {options}. Do not reply anything else other than 0/1/2/3. Do not explain your answer.

ctx:
{ctx}

options:
{options}
''' 

Chatbot_template = medi_template

# llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")

tokenizer = LlamaTokenizer.from_pretrained(os.getenv("MODEL_NAME"))
local_llm = AutoModelForCausalLM.from_pretrained(
    os.getenv("MODEL_NAME"), trust_remote_code=True,
    load_in_8bit=False,
    device_map='auto',
)
local_llm.eval()

pipe = pipeline(
    "text-generation",
    model=local_llm, 
    tokenizer=tokenizer, 
    max_length=2000,
    temperature=0,
    top_p=1
)
print("Model Loaded\n Model Name: ", local_llm)

llm = HuggingFacePipeline(pipeline=pipe)


qa_prompt = PromptTemplate(template=Chatbot_template, input_variables=["options","ctx"])

chatbot_chain = LLMChain(
    llm = llm,
    prompt = qa_prompt
)

df = pd.read_csv("csv/hellaswag_medi.csv")
# print("Dataset Loaded: ", df)
pred = []
actual = []
for i in range(5):
    print("Entered loop")
    context = df["ctx"][i].replace('"','')
    print(context)
    opt = df["cleaned_endings"][i]
    actual.append(df["label"][i])
    reply = chatbot_chain.predict(ctx=context,options=opt)
    print(reply)
    rep = int(reply)
    pred.append(rep)
    print(i)

result_df = pd.DataFrame(list(zip(actual, pred)),
               columns =['actual', 'predicted'])

result_df.to_csv("hellaswag_openai_try_3.csv")
print(accuracy_score(actual, pred))
