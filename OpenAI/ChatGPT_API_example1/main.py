from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def generateCatName():
    llm = OpenAI(temperature=0.8)

    names = llm.invoke("You have a new kitten and would like to give it a nice name. Give me a list of 5 possible names.")

    return names

if __name__ == "__main__":
    print(generateCatName())