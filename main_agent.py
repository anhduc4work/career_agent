
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
import os


class AgentState(MessagesState):
    cv: str
    jd: str
    sender: str
    new_cv: str
    chat_history_summary: str

from langchain_core.messages import RemoveMessage
memo_instruction = """You are summarization expert. Combine the current summary and the given conversation into only brief summary.
Remember to keep new summary short, brief in about 10-40 words, as short as possible.
Here is the current summarization (it can be empty):
{current_memo}
Here is the conversation to sum up:
{conversation}"""
def filter_and_save_messages(state, config, store):
    all_messages = state["messages"]
    if len(all_messages) >= 4:
        print("--filter--")
        # Save all but the 2 most recent messages to store
        messages_to_remove = all_messages[:-2]

        print("-save-")

        user_id = config["configurable"].get("user_id","")
        # 1, if there is user_id, personalize
        if user_id:
            namespace = ("chat_history", user_id)
            for m in messages_to_remove:
                store.put(namespace, m.id, {"data": m.type+ ": "+ m.content })
        
        # 2, Summarize messages:
        print("-sum-")

        current_summary = state.get("chat_history_summary", "")
        model = ChatTogether(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            temperature=0,
            api_key=os.environ["TOGETHER_API_KEY"]
        )
        updated_summary = model.invoke([SystemMessage(memo_instruction.format(
            current_memo = current_summary,
            conversation = messages_to_remove
        )), HumanMessage("Do it")])

        print("-del-")

        # 3, Delete all but the 2 most recent messages
        delete_messages = [RemoveMessage(id=m.id) for m in messages_to_remove]
        return {"messages": delete_messages, "chat_history_summary": updated_summary.content}
    else:
        # just pass
        return 

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
    If you have memory for this user, use it to personalize your responses.
    Here is the memory (it may be empty): {memory}
    """.strip()


def main_agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.
    """
    print("---CALL AGENT---")        
    messages = state["messages"]

    # handle error
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
                
    
    model = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.environ["OPENAI_API_KEY"]) 
    model = model.bind_tools(tools)
    response = model.invoke([SystemMessage(agent_instruction.format(
        tool_names = "\n".join([tool.name + ": " + tool.description for tool in tools]), 
        cv = state.get("cv", ""),
        jd = state.get("jd", ""),
        memory = state.get("chat_history_summary", "")))] + messages)
    return {"messages": [response], "sender": "agent"}


# build graph
def career_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("filter", filter_and_save_messages)
    workflow.add_node("agent", main_agent)
    workflow.add_node('tools', tool_node)
    workflow.set_entry_point("filter")
    workflow.add_edge("filter", "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition, 
    )
    workflow.add_edge("tools", "agent")

    ##### Checkpoint
    from langgraph.checkpoint.mongodb import MongoDBSaver
    import pymongo

    mongodb_client = pymongo.MongoClient(os.environ["MONGO_URI"])
    memory = MongoDBSaver(mongodb_client, db_name="scholarship", collection_name="checkpoints")

    ##### Store/Memory Longterm
    from langgraph.store.postgres import PostgresStore
    from psycopg import connect
    from langchain_ollama import OllamaEmbeddings

    conn = connect("dbname=postgres user=anhduc213 password=200103 host=localhost port=5432", autocommit=True)
    postgres_store = PostgresStore(conn, index={
            "embed": OllamaEmbeddings(model = 'nomic-embed-text'),
            "dims": 768,
        })
    postgres_store.setup()

    ##### Graph
    react_graph_memory = workflow.compile(checkpointer=memory, store=postgres_store)
    return react_graph_memory