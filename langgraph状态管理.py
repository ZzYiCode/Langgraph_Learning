from pydantic import Field, BaseModel
from typing import Optional
from langgraph.graph import StateGraph

class TaskState(BaseModel):
    user_query : str
    tool_result : Optional[str] = Field(default= None,description="工具调用结果")
    final_answer : Optional[str] = Field(default= None,description="最终答案")
    progress : Optional[int] = Field(default= None,description="任务进度百分比")


def parse_query(state: TaskState):
    print("="*10+"解析状态"+"="*10)
    print(state)
    query = state.user_query
    print("问题:" + query)
    update = {
        "user_query": query,
        "progress" : 30
    }
    return update

def call_tool(state: TaskState):
    print("=" * 10 + "工具调用" + "=" * 10)
    print(state)
    result = f"结果：关于{state.user_query}的相关知识"
    update = {
        "tool_result": result,
        "progress" : 60
    }
    return update

def generate_ans(state: TaskState):
    print("=" * 10 + "生成答案" + "=" * 10)
    print(state)
    tool_result = state.tool_result
    user_query = state.user_query
    ans = f"跟据工具结果：{tool_result},\n回答用户问题{user_query}"

    update = {
        "final_answer": ans,
        "progress" : 100
    }
    return update

builder = StateGraph(TaskState)

builder.add_node("parse_query", parse_query)
builder.add_node("call_tool", call_tool)
builder.add_node("generate_ans", generate_ans)

builder.set_entry_point("parse_query")
builder.add_edge("parse_query","call_tool")
builder.add_edge("call_tool","generate_ans")
graph = builder.compile()

init_state = TaskState(user_query="什么是 LangGraph？")

final_state = graph.invoke(init_state)

print("\n最终状态：")
print(final_state)