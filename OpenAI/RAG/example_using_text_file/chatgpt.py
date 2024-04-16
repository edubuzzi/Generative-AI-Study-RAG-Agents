from dotenv import load_dotenv
from langchain_openai import OpenAI
from tools import pre_process

load_dotenv()

llm = OpenAI(temperature=0.5, model="gpt-3.5-turbo-instruct")

if __name__ == "__main__":
    pre_process()