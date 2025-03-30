<!--
 * @Author: javapub iswangshiyu@foxmail.com
 * @Date: 2025-03-30 14:47:13
 * @LastEditors: javapub iswangshiyu@foxmail.com
 * @LastEditTime: 2025-03-30 14:54:59
 * @FilePath: /langchain-Demo/README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# LangChain 练习项目

这个仓库包含了LangChain框架的各种练习和示例代码。

## 环境设置

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/langchain-Demo.git
cd langchain-Demo
```

2. 创建并激活Python虚拟环境：
```bash
python3 -m venv venv
source venv/bin/activate  # 在Windows上使用: venv\Scripts\activate
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 设置环境变量：
```bash
cp .env.example .env
```
然后编辑`.env`文件，添加你的API密钥。

## 示例运行

运行Hello LangChain示例：
```bash
python hello_langchain.py
```

## 项目结构

- `hello_langchain.py` - 简单的LangChain示例
- `requirements.txt` - 项目依赖
- `.env.example` - 环境变量示例文件

## 学习资源

- [LangChain官方文档](https://python.langchain.com/docs/get_started/introduction)
- [LangChain GitHub仓库](https://github.com/langchain-ai/langchain)
