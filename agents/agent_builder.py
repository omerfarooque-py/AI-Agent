from langchain_classic.agents import Tool, AgentExecutor, create_react_agent
from langchain_classic import hub


def build_agent(llm, pdf_func, web_func):

    tools = [
        Tool(
            name="PDF_Search",
            func=pdf_func,
            description="Search PDF knowledge base first."
        ),
        Tool(
            name="Web_Search",
            func=web_func,
            description="Search the internet."
        )
    ]

    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=3,
        early_stopping_method="force"
    )