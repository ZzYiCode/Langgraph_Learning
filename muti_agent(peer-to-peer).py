from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph,START,END
from pydantic import BaseModel, Field
from typing import Literal
from config import LLM_BASE_URL,LLM_API_KEY,LLM_MODEL_ID

model = init_chat_model(
    model = LLM_MODEL_ID,
    api_key = LLM_API_KEY,
    base_url = LLM_BASE_URL,
)

class TeamState(BaseModel):
    project_goal : str = Field(description="项目目标")
    todo_tasks : list[str] = Field(default = [],description="待办事项，全体智能体共享可见")
    done_tasks : list[str] = Field(default = [],description="已办事项，全体智能体共享可见")
    status_updates : list[str] = Field(default = [],description="状态更新记录，智能体执行后记录，用于团队信息同步")
    is_finished : bool = Field(default=False,description="若为True则流程终止🛑")

AGENT_SKILLS = {
    "产品智能体": "负责梳理产品需求、设计MVP功能、输出产品文档，确保产品方向匹配项目目标",
    "研发智能体": "负责根据产品文档实现MVP代码、解决技术问题、保证功能可运行，输出可测试的产品",
    "运营智能体": "负责根据MVP设计推广方案、撰写推广文案、初步落地推广，带来种子用户"
}
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    你是创业团队的{agent_name}，核心技能是：{agent_skill}。
    团队无主管，所有人平等，你需基于全局项目状态自主判断是否执行任务，判断规则：
    1. 优先看「状态更新记录」：有新状态变化且需要你的技能衔接，必须主动干活；
    2. 再看「待办事项」：有待办且属于你的技能范围，主动认领执行；
    3. 最后看「项目目标」：无待办但目标未完成，主动提出待办并执行。

    ⚠️ 强制输出格式（必须严格遵守，缺一不可，用===分隔3个部分，不能合并、不能省略）：
    决策：执行/不执行
    ===
    原因：具体判断依据（基于全局状态的细节，不能简略）
    ===
    执行内容：执行则写具体做的事；不执行则严格写「无」，不能写其他内容

    ⚠️ 格式示例（必须按此结构输出）：
    决策：执行
    ===
    原因：状态更新显示产品完成了需求梳理，待办中有开发MVP代码的任务，属于我的研发技能范围
    ===
    执行内容：根据产品需求文档，实现AI智能体工具MVP的核心Python代码，完成本地功能测试
    """),
    # 用户输入部分：将全局状态的所有字段传入，让LLM基于完整状态决策
    ("user", "全局项目状态：\n项目目标：{project_goal}\n待办任务：{todo_tasks}\n已完成任务：{done_tasks}\n最新状态更新：{status_updates}\n项目是否完成：{is_finished}")
])

def agent_node(agent_name,agent_skill):


    def node(state : TeamState) -> TeamState:
        chain = prompt | model | StrOutputParser()
        res = chain.invoke(
            {
                "agent_name": agent_name,
                "agent_skill": agent_skill,
                ** state.model_dump()
            }
        )

        print(f"\n===== {agent_name} 原始返回 =====")
        print(res)
        print(f"=========================\n")

        parts = [p.strip() for p in res.split("===")]
        if len(parts) < 3:
            parts += ["决策：不执行", "原因：模型返回格式错误，兜底判定", "执行内容：无"][len(parts):]
        if len(parts) > 3:
            parts = parts[:3]
        try:
            # 解包为决策、原因、执行内容三部分
            decision_part, reason_part, action_part = parts
            # 提取核心内容：移除前缀（如决策：），处理模型可能的多余文字
            decision = decision_part.replace("决策：", "").strip() if "决策：" in decision_part else "不执行"
            reason = reason_part.replace("原因：", "").strip() if "原因：" in reason_part else "格式错误，兜底不执行"
            action = action_part.replace("执行内容：", "").strip() if "执行内容：" in action_part else "无"
            # 强制校验决策值：仅允许「执行/不执行」，其他值兜底为不执行（避免无效决策）
            if decision not in ["执行", "不执行"]:
                decision = "不执行"
                reason = f"决策值异常（{decision}），兜底判定不执行"
        except Exception as e:
            # 捕获所有解包/格式异常（如索引错误、类型错误），全部兜底为「不执行」
            decision = "不执行"
            reason = f"格式解析失败：{str(e)}，兜底判定不执行"
            action = "无"
        print(f"===== {agent_name} 标准化决策 =====")
        print(f"是否执行：{decision}")
        print(f"判断原因：{reason}")
        print(f"执行内容：{action}\n")

        if decision == "执行" and action != "无" and action != "「无」":
            new_state = state.model_copy()
            new_state.done_tasks = list(new_state.done_tasks) + [action]
            # ② 从「待办任务列表」移除已执行的任务（模糊匹配，避免文字完全一致的要求）
            new_state.todo_tasks = [t for t in new_state.todo_tasks if not any(k in t for k in action.split("：")[0].split("，"))]
            # ③ 添加状态更新记录：让其他智能体感知「该智能体完成了什么」，实现团队信息同步
            new_state.status_updates = list(new_state.status_updates) + [f"{agent_name}：{action}"]
            return new_state
        return state

    return node


product_agent = agent_node("产品智能体", AGENT_SKILLS["产品智能体"])
dev_agent = agent_node("研发智能体", AGENT_SKILLS["研发智能体"])
ops_agent = agent_node("运营智能体", AGENT_SKILLS["运营智能体"])

builder = StateGraph(TeamState)

builder.add_node("product", product_agent)  # 产品智能体节点
builder.add_node("dev", dev_agent)          # 研发智能体节点
builder.add_node("ops", ops_agent)          # 运营智能体节点

def should_continue(state: TeamState) -> Literal["product", "dev", "ops", END]:
    """
    LangGraph条件路由函数：运营智能体执行后，判断流程**继续循环**还是**终止**
    返回值约束为字面量：继续则返回下一个节点（product），终止则返回END
    :param state: 当前的全局状态
    :return: 下一个节点名称 / END（终止）
    """
    # 终止条件：① 手动置为项目完成 ② 状态更新数>3（完成3个核心任务，模拟项目结束）
    if state.is_finished or len(state.status_updates) > 3:
        return END  # 返回END，流程终止
    return "product"  # 未终止则返回产品智能体，继续循环执行（产品→研发→运营）

builder.set_entry_point("product")

builder.add_edge("product", "dev")
builder.add_edge("dev", "ops")
builder.add_conditional_edges("ops", should_continue)
graph = builder.compile()

if __name__ == "__main__":
    initial_state = TeamState(
        # 项目核心目标（固定不变）
        project_goal="开发一个AI智能体工具的MVP并完成初步推广，实现种子用户获取",
        # 初始待办任务（智能体自主认领执行，执行后自动移除）
        todo_tasks=[
            "梳理AI智能体工具MVP的核心需求",
            "实现MVP的核心功能代码",
            "撰写MVP推广文案并在小红书初步发布"
        ],
        done_tasks=[],  # 初始无已完成任务
        # 初始状态更新：标记项目启动，让所有智能体感知项目开始
        status_updates=["项目启动：开始推进AI智能体工具MVP开发与推广"],
        is_finished=False  # 初始项目未完成
    )

    # 打印项目启动信息，直观查看初始目标和待办
    print("===== 创业团队项目启动 =====")
    print(f"项目目标：{initial_state.project_goal}")
    print(f"初始待办：{initial_state.todo_tasks}\n")

    # 流式运行状态机：逐节点输出执行过程，直观看到智能体决策和状态更新
    # graph.stream()返回生成器，每次yield一个节点的执行结果（节点名称+更新后的状态）
    for step in graph.stream(initial_state):
        # 遍历每一步的节点和状态（单节点执行，故仅一个键值对）
        for node, state in step.items():
            print(f"===== 节点 {node} 执行后 - 全局状态 =====")
            print(f"✅ 已完成任务：{state['done_tasks']}")
            print(f"📋 剩余待办任务：{state['todo_tasks']}")
            print(f"📌 最新团队状态：{state['status_updates'][-1]}\n")
