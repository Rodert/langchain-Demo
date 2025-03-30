from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# 初始化 Ollama 模型
llm = OllamaLLM(model="llama3:latest")

# 创建提示模板
template = PromptTemplate.from_template(
    "以下是一个用户的问题，请你必须使用中文回答：\n\n{question}\n\n记住：必须用中文回答！"
)

# 格式化提示
prompt = template.format(question="你好，请介绍一下自己")

# 发送请求
response = llm.invoke(prompt)

print(response)