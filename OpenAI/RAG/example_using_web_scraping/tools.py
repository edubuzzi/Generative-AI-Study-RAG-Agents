import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning

def pre_process():
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)