from langchain.output_parsers import PydanticOutputParser, CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, Field
from typing import List

def comma_separated_parser():
    """逗号分隔列表解析器示例"""
    print("=== 逗号分隔列表解析器 ===\n")
    
    # 创建解析器
    parser = CommaSeparatedListOutputParser()
    
    # 获取格式说明
    format_instructions = parser.get_format_instructions()
    print("格式说明:", format_instructions)
    
    # 创建提示模板
    template = """请列出{topic}的五个例子。
    
{format_instructions}"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["topic"],
        partial_variables={"format_instructions": format_instructions}
    )
    
    # 填充模板
    formatted_prompt = prompt.format(topic="中国的传统节日")
    print("\n格式化提示:")
    print(formatted_prompt)
    
    # 使用模型和解析器
    llm = OllamaLLM(model="llama3:latest")
    
    # 链式调用
    chain = llm | parser
    
    # 执行
    result = chain.invoke(formatted_prompt)
    
    print("\n解析结果 (列表):")
    print(result)
    print(f"类型: {type(result)}")
    print("\n" + "-"*50 + "\n")

def pydantic_parser():
    """Pydantic 解析器示例"""
    print("=== Pydantic 解析器 ===\n")
    
    # 定义输出模型
    class Movie(BaseModel):
        title: str = Field(description="电影标题")
        director: str = Field(description="导演姓名")
        year: int = Field(description="发行年份")
        rating: float = Field(description="评分 (1-10)")
        genres: List[str] = Field(description="电影类型列表")
    
    # 创建解析器
    parser = PydanticOutputParser(pydantic_object=Movie)
    
    # 获取格式说明
    format_instructions = parser.get_format_instructions()
    print("格式说明:")
    print(format_instructions)
    
    # 创建提示模板
    template = """请提供一部著名的{country}电影信息。
    
{format_instructions}"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["country"],
        partial_variables={"format_instructions": format_instructions}
    )
    
    # 填充模板
    formatted_prompt = prompt.format(country="中国")
    print("\n格式化提示:")
    print(formatted_prompt)
    
    # 使用模型和解析器
    llm = OllamaLLM(model="llama3:latest")
    
    # 链式调用
    chain = llm | parser
    
    try:
        # 执行
        result = chain.invoke(formatted_prompt)
        
        print("\n解析结果 (Pydantic 对象):")
        print(f"标题: {result.title}")
        print(f"导演: {result.director}")
        print(f"年份: {result.year}")
        print(f"评分: {result.rating}")
        print(f"类型: {result.genres}")
        print(f"\n完整对象: {result.model_dump()}")
    except Exception as e:
        print(f"解析错误: {e}")
        print("原始输出:")
        raw_output = llm.invoke(formatted_prompt)
        print(raw_output)
    
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    # 测试逗号分隔列表解析器
    comma_separated_parser()
    
    # 测试 Pydantic 解析器
    pydantic_parser() 