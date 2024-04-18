from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from tools import pre_process

load_dotenv()

def oscar(movie, year, llm):
    prompt = PromptTemplate(
        input_variables=["movie", "year"],
        template="How many oscars {movie} won in {year}"
    )

    oscar_chain = LLMChain(llm=llm, prompt=prompt)

    response = oscar_chain({"movie": movie, "year": year})

    return response

llm = OpenAI(temperature=0.5, model="gpt-3.5-turbo-instruct")

if __name__ == "__main__":
    pre_process()
    response = oscar("Oppenheimer", 2024, llm)
    print(response["text"])