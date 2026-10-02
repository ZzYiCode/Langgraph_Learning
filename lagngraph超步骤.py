import keyword
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph,START,END
from pydantic import BaseModel,Field
from typing import Optional
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()



model = init_chat_model(
    model = "deepseek-v4-flash",
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = os.getenv("LLM_BASE_URL"),
)
summary_prompt = ChatPromptTemplate.from_template(
    template="{text}\n将以上文本提取摘要，50字以内"
)
keyword_prompt = ChatPromptTemplate.from_template(
    template="{text}\n将以上文本提取关键词，5个以内"
)
summary_chain = summary_prompt | model | StrOutputParser()
keyword_chain = keyword_prompt | model | StrOutputParser()



class TaskState(BaseModel):
    raw_text: str
    summary: Optional[str] = Field(default=None,description="文本摘要")
    keyword : Optional[str] = Field(default=None,description="文本关键词")
    has_finished: bool = Field(default=False,description="是否完成")
    final_answer : Optional[str] = Field(default=None,description="最终答案")


def summary_node(state : TaskState):
    text = state.raw_text
    res = ""
    if text:
        res = summary_chain.invoke({"text": text})
    print("✅️成功提取摘要")
    update = {
        "summary": res,
    }
    return update

def keyword_node(state : TaskState):
    text = state.raw_text
    res = ""
    if text:
        res = keyword_chain.invoke({"text": text})
    print("✅️成功提取关键词")
    update = {
        "keyword": res,
    }
    return update

def check_node(state : TaskState):
    has_finish = False
    if state.summary and state.keyword :
        print("✅️通过检查")
        print(state.summary)
        print(state.keyword)
        has_finish = True

    update = {
        "has_finished": has_finish,
    }
    return update

def route_by_finished(state : TaskState):
    if state.has_finished==True:
        return "generate_ans"
    else:
        return "error_node"



def error_node(state : TaskState):
    print("❌️流程执行失败")




def generate_ans(state : TaskState):
    final_answer = f"{state.summary}\n{state.keyword}"
    update = {
        "final_answer": final_answer,
    }
    return update


builder = StateGraph(TaskState)

builder.add_node("summary_node", summary_node)
builder.add_node("keyword_node", keyword_node)
builder.add_node("check_node", check_node)
builder.add_node("error_node", error_node)
builder.add_node("generate_ans", generate_ans)

builder.add_edge(START,"summary_node")
builder.add_edge(START,"keyword_node")
builder.add_edge("summary_node","check_node")
builder.add_edge("keyword_node","check_node")

builder.add_conditional_edges(
    "check_node",
    route_by_finished,
    {
        "generate_ans": "generate_ans",
        "error_node": "error_node",
    }
)

graph = builder.compile()


text = """
    南郑区气象台2026年09月30日14时36分发布大风蓝色预警信号：预计下述地区未来24小时内将受大风影响，平均风力可达6级以上（阵风7级以上）：小南海镇，请注意防范。

防御指南：

1.政府及相关部门按照职责做好防大风工作；
2.关好门窗，加固围板、棚架、广告牌等易被风吹动的搭建物，妥善安置易受大风影响的室外物品，遮盖建筑物资；

3. 相关水域水上作业和过往船舶采取积极的应对措施，如回港避风或者绕道航行等；

4.行人注意尽量少骑自行车，刮风时不要在广告牌、临时搭建物等下面逗留；

5.有关部门和单位注意森林、草原等防火。
"""

print(graph.invoke({"raw_text" : text}))

