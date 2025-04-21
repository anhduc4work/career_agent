import os
import json
import dotenv
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.types import Command
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage, SystemMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_ollama import OllamaEmbeddings
from psycopg import connect

dotenv.load_dotenv()

class AgentState(MessagesState):
    cv: str
    jd: str
    sender: str
    new_cv: str
    chat_history_summary: str = ""
    last_index: int = 0

class CareerAgent:
    def __init__(self, tools, pg_uri: str):
        self.tools = tools
        self.pg_uri = pg_uri
        self.checkpointer = None
        self.store = None
        self.graph = None
        self.tool_node = ToolNode(
            self.tools,
            name="tools",
            handle_tool_errors=self._handle_error
        )

    def _handle_error(self, error="") -> str:
        return json.dumps({'error': str(error)}) if error else json.dumps({'error': "unknown error"})

    def _filter_and_save_messages(self, state, config, store):
        all_messages = state["messages"]
        last_index = state.get("last_index") or 0
        not_sum_messages = all_messages[last_index:]

        WINDOWSIZE = 6
        MINNEWMESSAGESALLOW = 4

        if len(not_sum_messages) >= MINNEWMESSAGESALLOW + WINDOWSIZE:
            messages_to_sum = not_sum_messages[:-(WINDOWSIZE)]
            new_last_index = last_index + len(messages_to_sum)

            print("-sum-thread-")
            current_summary = state.get("chat_history_summary", "")
            model = ChatTogether(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                temperature=0,
                api_key=os.environ["TOGETHER_API_KEY"]
            )
            updated_summary = model.invoke([
                SystemMessage(self.memo_instruction.format(
                    current_memo=current_summary,
                    conversation=messages_to_sum
                )),
                HumanMessage("Do it")
            ])

            print("-save_thread_db-")
            user_id = config["configurable"].get("user_id", "")
            if user_id:
                namespace = ("chat_history", user_id)
                for m in messages_to_sum:
                    store.put(namespace, m.id, {"data": m.type + ": " + m.content})
                store.put(namespace, "user_info", {"data": updated_summary.content})

            return Command(update={"chat_history_summary": updated_summary.content, "last_index": new_last_index})

        return None

    def _main_agent(self, state, config: RunnableConfig, store):
        print("---CALL AGENT---")
        last_index = state.get("last_index", 0)
        messages = state["messages"][last_index:]
        user_id = config["configurable"].get("user_id", "")
        namespace = ("chat_history", user_id)
        config["recursion_limit"] = 2

        if not messages:
            return Command(update={"messages": [AIMessage("No message provided.")]}, goto=END)

        if isinstance(messages[-1], ToolMessage):
            try:
                error = json.loads(messages[-1].content).get("error", "")
                if error:
                    return Command(update={"messages": [AIMessage(error)]}, goto=END)
            except Exception:
                pass

        print("memo: ", state.get("chat_history_summary", ""))
        print("user_info: ", store.get(namespace, "user_info") or "")

        model = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.environ["OPENAI_API_KEY"])
        model = model.bind_tools(self.tools)

        user_info = ""
        if store.get(namespace, "user_info"):
            user_info = store.get(namespace, "user_info").value["data"]

        response = model.invoke([
            SystemMessage(self.agent_instruction.format(
                tool_names="\n".join([tool.name + ": " + tool.description for tool in self.tools]),
                cv=state.get("cv", ""),
                jd=state.get("jd", ""),
                user_info=user_info,
                thread_memory=state.get("chat_history_summary", "")
            ))
        ] + messages)

        return Command(update={"messages": [response], "sender": "agent"})

    def setup_memory_and_store(self):
        conn = connect(self.pg_uri, autocommit=True)
        self.store = PostgresStore(conn, index={
            "embed": OllamaEmbeddings(model='nomic-embed-text'),
            "dims": 768,
        })
        self.store.setup()

        self.checkpointer = PostgresSaver(conn)
        self.checkpointer.setup()

    def build(self):
        self.memo_instruction = """You are summarization expert. Combine the current summary and the given conversation into only brief summary.
Remember to keep new summary short, brief in about 10-40 words, as short as possible.
Here is the current summarization (it can be empty):
{current_memo}
Here is the conversation to sum up:
{conversation}"""

        self.agent_instruction = """You are a helpful AI assistant, collaborating with other assistants.
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
Here is your summary of recent chat with user: {thread_memory}
Here is your memory (it may be empty): {user_info}
If you have memory for this user, use it to personalize your responses.
""".strip()

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", lambda state, config: self._main_agent(state, config, self.store))
        workflow.add_node("filter", lambda state, config: self._filter_and_save_messages(state, config, self.store))
        workflow.add_node("tools", self.tool_node)

        workflow.set_entry_point("agent")
        workflow.add_edge("agent", "filter")
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_edge("tools", "agent")

        self.graph = workflow.compile(checkpointer=self.checkpointer, store=self.store)

    def get_graph(self):
        return self.graph
