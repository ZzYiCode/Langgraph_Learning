import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()
model = init_chat_model(
    model = "deepseek-v4-flash",
    api_key = os.getenv("LLM_API_KEY"),
    base_url = os.getenv("LLM_BASE_URL"),
)

extract_sell_prompt = ChatPromptTemplate.from_template(
    template = "请根据以下描述，简练地提取出该产品的核心卖点（3个）\n{info}",
    # input_variables = ["info"],
)
extract_people_prompt = ChatPromptTemplate.from_template(
    template = "请根据以下描述，简练地提取出该产品的核心人群\n{info}",
    # input_variables = ["info"],
)
market_prompt = ChatPromptTemplate.from_template(
    template = """请根据该产品的核心卖点和核心人群，写一段吸引消费者的营销话术.\n
                核心卖点:{sell}\n核心人群:{people}""",
    # input_variables = ["sell", "people"],
)
def print_prompt(input):
    print(input)
    return input
chain = (
    {
        "people" : extract_people_prompt | model | StrOutputParser()|RunnableLambda(print_prompt),
        "sell" : extract_sell_prompt | model | StrOutputParser()|RunnableLambda(print_prompt),
    } | market_prompt | model | StrOutputParser()
)

product_intro = """这款无线耳机采用蓝牙5.3芯片，连接稳定无延迟，支持高清通话；续航长达30小时，充电10分钟可使用2小时；机身采用亲肤硅胶材质，佩戴舒适，防水防汗，适合运动使用。"""

res = chain.invoke(product_intro)

print(res)
