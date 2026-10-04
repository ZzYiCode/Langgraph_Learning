from typing import Literal
from langgraph.graph import StateGraph,START,END
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from load_prompt import construction_prompt,write_prompt
from config import LLM_BASE_URL, LLM_API_KEY

class TaskState(BaseModel):
    content : str = ""
    title : str = Field(default="",description="小说标题")
    character : str = Field(default="",description="小说主人公")
    description : str = Field(default="",description="大致情节")
    construction :str = Field("",description="故事框架")
    review_text : str = Field("",description="指导意见")
    progress : Literal[START,"construction","write",END] = "construction"
    is_finished : bool = False


model = init_chat_model(
    model = "deepseek-v4-flash",
    api_key = LLM_API_KEY,
    base_url = LLM_BASE_URL,
)

def construction_node(state: TaskState):
    review_text = state.review_text or "暂无指导意见"
    prompt = ChatPromptTemplate.from_template(
        template_format = construction_prompt
    )
    chain = prompt | model | StrOutputParser()
    result = chain.invoke(
        {
            "description": state.description,
            "review_text": review_text,
            "character": state.character,
            "title": state.title,
        }
    )
    return {
        "progress" : "construction",
        "construction": result,
        "review_text": None
    }

def writer_node(state: TaskState):
    review_text = state.review_text or "暂无指导意见"
    prompt = ChatPromptTemplate.from_template(
        template_format = write_prompt
    )
    chain = prompt | model | StrOutputParser()
    result = chain.invoke(
        {
            "description": state.description,
            "construction": state.construction,
            "review_text": review_text,
            "character": state.character,
            "title": state.title,
        }
    )
    return {
        "progress" : "write",
        "construction": result,
        "review_text": None
    }

def review_node(state: TaskState):
    now_progress = state.progress


