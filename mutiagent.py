import os
from typing import Optional
from pydantic import BaseModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


load_dotenv()

model = init_chat_model(
    model = "deepseek-v4-flash",
    api_key = os.getenv("LLM_API_KEY"),
    base_url = os.getenv("LLM_BASE_URL"),
)

class TaskState(BaseModel):
    task : str
    research : Optional[str] = None
    draft : Optional[str] = None
    code : Optional[str] = None
    math : Optional[str] = None
    next_agent : Optional[str] = None
    result : Optional[str] = None
    round_count : int = 0
    supervisor_thoughts : Optional[str] = None

MAX_ROUND = 3
research_agent = ChatPromptTemplate.from_messages([
    ("user", "请调研以下任务的背景信息，整理成条列要点，中文输出：{task}")
]) | model

writer_agent = ChatPromptTemplate.from_messages([
    ("user", "根据以下信息撰写中文技术文章或说明文：{research}")
]) | model

code_agent = ChatPromptTemplate.from_messages([
    ("user", "请根据以下任务生成 Python 示例代码：{task}")
]) | model

math_agent = ChatPromptTemplate.from_messages([
    ("user", "请解决以下数学/逻辑问题，并详细说明过程：{task}")
]) | model

def supervisor_node(state : TaskState ):
    new_round = state.round_count + 1

    if(new_round > MAX_ROUND):
        print("超过最大次数——>结束任务🔚")
        return {
            "round_count" : new_round,
            "next_agent" : "end",
            "supervisor_thoughts" : "超过轮次上限，结束任务"
        }
    prompt = f"""
    你是多智能体系统的主管智能体（Supervisor），负责调度专家智能体，但你不执行任务。请阅读当前任务和已完成状态，并选择下一步最合适的智能体执行。
    
    任务：
    {state.task}
    
    已完成状态：
    - 调研: {"已完成" if state.research else "未完成"}
    - 写作: {"已完成" if state.draft else "未完成"}
    - 编程: {"已完成" if state.code else "未完成"}
    - 数学: {"已完成" if state.math else "未完成"}
    
    可调度智能体：
    - research_agent：负责调研和整理资料
    - writer_agent：负责撰写中文文章或说明文
    - code_agent：负责编写 Python 代码
    - math_agent：负责数学/逻辑计算与推理
    
    约束：
    1. 不能选择已完成的智能体。
    2. 必须选择与任务相关的智能体。
    3. 如果所有任务完成，返回 "end"。
    4. 请在回答中先写出你的“思考过程”，然后在最后一行返回下一步智能体名称（research_agent / writer_agent / code_agent / math_agent / end）。
    
    请用中文完整回答：
    """
    supervisor_prompt = ChatPromptTemplate.from_template(
        template = prompt,
    )
    supervisor_agent = supervisor_prompt | model | StrOutputParser()
    res = supervisor_agent.invoke({"task" : state.task}).strip()

    last_line = res.split("\n")[-1]
    valid_agents = ("research_agent", "writer_agent", "code_agent", "math_agent", "end")
    next_agent = next((a for a in valid_agents if a in last_line), "end")
    supervisor_thoughts = res
    print(f"🧠 主管思考过程：\n{res}\n")
    print(f"🧠 主管调度 → {next_agent} (轮次 {new_round})")
    return {
        "round_count" : state.round_count,
        "next_agent" : next_agent,
        "supervisor_thoughts" : supervisor_thoughts
    }


def research_node(state: TaskState):
    print(">>> Research Agent 执行中...")
    try:
        res = research_agent.invoke({"task": state.task})
        result = res.content.strip()
    except Exception as e:
        result = f"调研失败：{str(e)[:50]}"

    return {
        "research": result,
        "result": result
    }


def writer_node(state: TaskState):
    print(">>> Writer Agent 执行中...")
    try:
        res = writer_agent.invoke({"research": state.research})
        result = res.content.strip()
    except Exception as e:
        result = f"写作失败：{str(e)[:50]}"

    return {
        "draft": result,
        "result": result
    }


def code_node(state: TaskState):
    print(">>> Code Agent 执行中...")
    try:
        res = code_agent.invoke({"task": state.task})
        result = res.content.strip()
    except Exception as e:
        result = f"代码生成失败：{str(e)[:50]}"

    return {
        "code": result,
        "result": result
    }


def math_node(state: TaskState):
    print(">>> Math Agent 执行中...")
    try:
        res = math_agent.invoke({"task": state.task})
        result = res.content.strip()
    except Exception as e:
        result = f"数学求解失败：{str(e)[:50]}"

    return {
        "math": result,
        "result": result
    }


workflow = StateGraph(TaskState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("writer_agent", writer_node)
workflow.add_node("code_agent", code_node)
workflow.add_node("math_agent", math_node)
workflow.add_node("research_agent", research_node)


workflow.add_edge(START, "supervisor")
workflow.add_edge("research_agent", "supervisor")
workflow.add_edge("writer_agent", "supervisor")
workflow.add_edge("code_agent", "supervisor")
workflow.add_edge("math_agent", "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    lambda state : state.next_agent,
    {
        "research_agent": "research_agent",
        "writer_agent": "writer_agent",
        "code_agent": "code_agent",
        "math_agent": "math_agent",
        "end": END
    }
)




graph = workflow.compile()

if __name__ == "__main__":
    tasks = [
        "撰写一篇介绍 LangGraph 多智能体协作的中文文章，面向初学者",
    ]

    for t in tasks:
        print("\n" + "="*50)
        print(f"任务：{t}")
        init_state = {
            "task": t,
            "research": None,
            "draft": None,
            "code": None,
            "math": None,
            "next_agent": None,
            "result": None,
            "round_count": 0,
            "supervisor_thoughts": None
        }
        result = graph.invoke(init_state)
        print("\n✅ 最终结果：\n", result["result"])