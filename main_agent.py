
from tools import tools
import json
from langgraph.prebuilt import ToolNode, tools_condition
import dotenv
dotenv.load_dotenv()

########## Tools ############
def _handle_error(error = "") -> str:
    if error:
        return json.dumps({'error': str(error)})
    else:
        return {'error': "unknown error"}
tool_node = ToolNode(tools, name="tools", handle_tool_errors = _handle_error)


from langchain_together import ChatTogether
from langchain_openai import ChatOpenAI

from langgraph.types import Command
from langgraph.graph import MessagesState, END, StateGraph
from langchain_core.messages import ToolMessage, SystemMessage, AIMessage, HumanMessage


class AgentState(MessagesState):
    cv: str
    jd: str
    sender: str
    new_cv: str


agent_instruction = """You are a helpful AI assistant, collaborating with other assistants.
    Use the provided tools to progress towards answering the question.
    If you are unable to fully answer, that's OK, another assistant with different tools 
    will help where you left off. Execute what you can to make progress.
    If you or any of the other assistants have the final answer or deliverable,
    prefix your response with FINAL ANSWER so the team knows to stop.
    You should return data in table markdown for easily interpretation (for task relating comparation)
    You have access to the following tools: 
    {tool_names}
    Here is the content of curriculum vitate of user (this is empty when user haven't uploaded it yet):
    {cv}
    Here is the content of job description that user mannually upload (this can be empty is they haven't upload):
    {jd}
    Note: if user tell you they already upload cv in messages, consider carefully if they do did it before calling tool 'update_cv_from_chat'
    """.strip()

def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.
    """
    print("---CALL AGENT---")        
    messages = state["messages"]

    # model = ChatTogether(
    #     model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    #     max_tokens= 2000,
    #     max_retries= 2, temperature=0,
    # )
    model = ChatOpenAI(model="gpt-4o", temperature=0) 
    model = model.bind_tools(tools)


    if isinstance(messages[-1], ToolMessage):
        try:
            error = json.loads(messages[-1].content).get("error", "")
            if error:
                return Command(
                    update = {"messages": [AIMessage(error)]},
                    goto = END
                )
        except Exception:
            pass
                
    
    response = model.invoke([SystemMessage(agent_instruction.format(
        tool_names = "\n".join([tool.name + ": " + tool.description for tool in tools]), 
        cv = state.get("cv", ""),
        jd = state.get("jd", "")))] + messages)

    return {"messages": [response], "sender": "agent"}

# build graph

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
workflow.add_node('tools', tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition, 
)
workflow.add_edge("tools", "agent")
from langgraph.checkpoint.mongodb import MongoDBSaver
import pymongo
import os

mongodb_client = pymongo.MongoClient(os.environ["MONGO_URI"])
memory = MongoDBSaver(mongodb_client, db_name="scholarship", collection_name="checkpoints")
react_graph_memory = workflow.compile(checkpointer=memory)