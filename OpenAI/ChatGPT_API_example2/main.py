from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

def generateCatName(animalType, animalColor):
    llmOpenAI = OpenAI(temperature=0.8)

    promptAnimalName = PromptTemplate(
        input_variables=["animal_type", "animal_color"],
        template="You have a new {animal_color} {animal_type} and would like to give him a nice name. Give me a list of 5 possible names."
    )
    animalNameChain = LLMChain(llm=llmOpenAI, prompt=promptAnimalName)
    return animalNameChain({"animal_type": animalType, "animal_color": animalColor})

if __name__ == "__main__":
    print(generateCatName())