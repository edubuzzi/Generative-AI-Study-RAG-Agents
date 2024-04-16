from langchain import hub
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import YouTubeSearchTool
from langchain_community.tools.google_trends import GoogleTrendsQueryRun
from langchain_community.utilities.google_trends import GoogleTrendsAPIWrapper
from langchain.agents import create_openai_functions_agent, AgentExecutor

load_dotenv()

# Tools
youtube_tool = YouTubeSearchTool()
google_trends = GoogleTrendsQueryRun(api_wrapper=GoogleTrendsAPIWrapper())

tools = [youtube_tool, google_trends]

# Prompt
prompt = hub.pull("hwchase17/openai-functions-agent")

# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Merge Tools, Prompt, LLM and Agent
agent = create_openai_functions_agent(llm, tools, prompt)

# Execute
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "Give me some YouTube links about llms."})