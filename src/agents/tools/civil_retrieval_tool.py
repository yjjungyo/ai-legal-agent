from src.agents.tools.retrieval_factory import make_fixed_retrieval_tool
def get_tool():
    return make_fixed_retrieval_tool("civil_retrieval_tool", "civil", "민법")
