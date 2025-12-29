from langchain_core.runnables import RunnableLambda

def parse_number(text:str) -> int:
    return int(text.strip())

parse_runnable = RunnableLambda(parse_number)

rumber = parse_runnable.invoke("  42 ")
print(rumber)