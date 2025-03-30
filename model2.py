'''
Author: javapub iswangshiyu@foxmail.com
Date: 2025-03-30 17:24:44
LastEditors: javapub iswangshiyu@foxmail.com
LastEditTime: 2025-03-30 17:26:07
FilePath: /langchain-Demo/model2.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage

# 初始化Ollama模型
llm = OllamaLLM(model="llama3:latest")  # 使用您已下载的模型

# 方法1: 使用 ChatPromptTemplate 生成消息列表
template = ChatPromptTemplate.from_messages([
    ("system", "你是一名{role}专家。"),
    ("human", "请回答以下问题：{question}")
])

# 填充模板生成消息列表
messages = template.format_messages(
    role="Python编程", 
    question="如何使用列表推导式？"
)

# 使用消息列表生成回答
response = llm.invoke(messages)
print("方法1结果:")
print(response)
print("\n" + "-"*50 + "\n")

# 方法2: 直接使用字符串提示
simple_prompt = "请用中文解释Python列表推导式，并给出三个例子。"
response2 = llm.invoke(simple_prompt)
print("方法2结果:")
print(response2)