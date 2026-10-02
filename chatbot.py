import os,re
from langchain_core.chat_history import InMemoryChatMessageHistory,BaseChatMessageHistory
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_experimental.tools import PythonREPLTool
from dotenv import load_dotenv
load_dotenv()
calc_tool = PythonREPLTool()

model = init_chat_model(
    model= "deepseek-v4-flash",
    api_key = os.getenv("LLM_API_KEY"),
    base_url = os.getenv("LLM_BASE_URL"),
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一名友好的个人助手，规则如下：
    1. 能记住最近{window_size}轮对话内容，用简单语言解答问题；
    2. 如果问题包含数学计算（如加减乘除、公式、数值运算），先调用计算工具得到结果，再用自然语言解释；
    3. 非计算问题直接回答，记得结合历史对话上下文。"""),
    MessagesPlaceholder(variable_name="chat_history"),  # 窗口记忆注入点
    ("human", "{input}")  # 用户新问题
])

def judge_and_clac(inputs):
    """
        核心逻辑：
        1. 检测用户问题是否包含数学计算需求
        2. 是：调用PythonREPLTool计算，再结合LLM生成回答
        3. 否：直接用LLM回答
    """
    user_input = inputs["input"]
    chat_history = inputs["chat_history"]

    calc_pattern = r"(\+|\-|\×|\*|÷|/|=|计算|求和|求差|平方|立方|多少|等于)"
    is_calc_needed = bool(re.search(calc_pattern, user_input))

    if is_calc_needed:
        try:
            calc_expr = re.sub(r"[^\d\+\-\*\/\(\)\.]", "", user_input)
            if not calc_expr:
                calc_result = "未识别可计算的表达式"
            else:
                calc_result = calc_tool.run(calc_expr)
        except Exception as e:
            calc_result = f"计算时出错\n{str(e)}"

        enhanced_input = f"""
               用户问题：{user_input}
               计算过程/结果：{calc_result}
               请结合计算结果，用简单易懂的语言回答用户问题，同时参考历史对话：{chat_history}
               """
        inputs["input"] = enhanced_input


    return inputs

window_memory_store = {}
WINDOW_SIZE = 2

def get_window_session_memory(session_id : str):

    if session_id not in window_memory_store:
        window_memory_store[session_id] = InMemoryChatMessageHistory()
    else:
        chat_history = window_memory_store[session_id]
        message_length = len(chat_history.messages)
        if message_length > 2*WINDOW_SIZE:
            chat_history = chat_history[-2*WINDOW_SIZE]
            return chat_history
        else:
            return chat_history
    return window_memory_store[session_id]

chain = RunnableLambda(judge_and_clac)| prompt | model

chain_with_window_memory = RunnableWithMessageHistory(
    runnable= chain,
    get_session_history= get_window_session_memory,
    input_messages_key= "input",
    history_messages_key= "chat_history",
    output_messages_key= "output",
)

if __name__ == "__main__":
    session_id = "user_001"

    while True:
        query = input("user:")
        if query in ["退出","quit"]:
            print("进程已结束")
            break
        else:
            res = chain_with_window_memory.invoke(
                {"input": query, "window_size": WINDOW_SIZE},
                config={"configurable" :{"session_id" : session_id}},
            )
            print(res.content)



