from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import ToolMessage, SystemMessage, AIMessage, HumanMessage
import uuid
import time
import gradio as gr
import dotenv
dotenv.load_dotenv()
from main_agent import CareerAgent
from tools import tools
import os

PG_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"
agent = CareerAgent(tools, PG_URI)
agent.setup_memory_and_store()
agent.build()
graph = agent.get_graph()



def process_file(cv_path):
    """Process link file pdf"""
    if not cv_path:
        return "No file uploaded!", gr.update(visible=True)
    
    loader = PyPDFLoader(cv_path)
    pages = loader.load_and_split()
    text = "\n".join([page.page_content for page in pages])
    
    return text  

def hide():
    """Hide component"""
    return gr.update(visible=False)

def show():
    """Show component"""
    return gr.update(visible=True)

def check_for_new_cv(config):
    """Check new CV"""
    return graph.get_state(config).values.get("new_cv", "Not available")

def check_for_thread_summary(config):
    """Check thread summary"""
    return graph.get_state(config).values.get("chat_history_summary", "Not available")

def check_for_user_info_cross_thread(config):
    """Check user info"""
    user_id = config["configurable"]["user_id"]
    namespace = ("user_info", user_id)
    user_info = graph.store.search(namespace)
    # for m in user_info:
    #     print(m.dict())
        
    if user_info:
        user_info = "\n".join(f"{idx+1}: {item.value['content']}" for idx, item in enumerate(user_info))
        return user_info
    else:
        return "Not Available"
    
def update_underthehood_ui(config):
    return check_for_new_cv(config), check_for_thread_summary(config), check_for_user_info_cross_thread(config)

def update_config_and_ui(thread_id, user_id):
    """Update interface to fit config (user_id and thread id)"""
    config = {"configurable":{"thread_id": thread_id, "user_id": user_id}}
    state = graph.get_state(config).values
    chat_history = []
    if state:
        for mess in state["messages"]:
            if isinstance(mess, HumanMessage):
                chat_history.append({"role": "user", "content": mess.content})
            elif isinstance(mess, AIMessage):
                chat_history.append({"role": "assistant", "content": mess.content})
            else:
                pass
        return chat_history, config, state.get("cv", "Not avaiable"), state.get("new_cv", "Not avalable")
    else:
        return "", config, "Not avaiable", "Not avaiable"


def user(user_message, chat_history):
    """Immediately adding user input to frontend chat"""

    print("-u-")
    if user_message["files"]:
        file_content = process_file(user_message["files"][0])
        return gr.MultimodalTextbox(value=None), chat_history + [
            {"role": "user", "content": user_message["text"], "metadata": {"id": str(uuid.uuid4())}},
            {"role": "user", "content": file_content, "metadata": {"title": "File included", "id": str(uuid.uuid4())}},            
            ]
    else:
        return gr.MultimodalTextbox(value=None), chat_history + [
            {"role": "user", "content": user_message["text"], "metadata": {"id": str(uuid.uuid4())}},
            ]

def bot(config, chat_history):
    """Bot Response, Stream token"""
    print("-b-")
    # if file
    last_message = chat_history[-1]
    if last_message["metadata"].get("title", ""):
        state = {
            "messages": [HumanMessage(chat_history[-2]["content"], id = chat_history[-2]["metadata"]["id"])],
            "cv": last_message["content"], 
        }
    else:
        state = {
            "messages": [HumanMessage(last_message["content"], id = last_message["metadata"]["id"])],
        }
    print(config)
    bot_message = graph.invoke(state, config)

    chat_history.append({"role": "assistant", "content": "", "metadata": {"id":bot_message["messages"][-1].id}})
    for character in bot_message["messages"][-1].content:
        chat_history[-1]['content'] += character
        time.sleep(0.001)
        yield chat_history

    chat_history[-1]['content'] = bot_message["messages"][-1].content
    return chat_history

def handle_edit(history, edit_data: gr.EditData):
    """Update frontend chat interface"""
    new_history = history[:edit_data.index]
    id = history[edit_data.index]["metadata"]["id"]
    new_history.append({"role": "user", "content": edit_data.value, "metadata": {"id": id}})
    return new_history

def fork_messages(config, chat_history):
    """Edit backend messages and return the ready state from history to invoke"""
    hist = graph.get_state_history(config)
    last_message = chat_history[-1]

    for i, s in enumerate(hist):
        if i%2==1:
            pass
        else:
            if s.values["messages"][-1].id == last_message["metadata"]["id"]:
                to_fork = s.config
                break

    # update
    fork_config = graph.update_state(to_fork,
                        {"messages": [HumanMessage(content=last_message["content"], 
                               id=last_message["metadata"]["id"])]},)
    
    fork_config["configurable"]["user_id"] = config["configurable"]["user_id"]
    return fork_config

def clear_checkpoint_in_config(config):
    """delete checkpoint id in config after forking"""
    if config["configurable"].get("checkpoint_id", ""):
        config["configurable"].pop("checkpoint_id")
    return config

def gen_id():
    """Gen new id"""
    print("---gen id---")
    new_id = str(uuid.uuid4())
    return gr.update(value=new_id)

def update_dropdown(choices, new_id = False):
    """Update choice when user id is change or add"""
    if not new_id:
        new_id = str(uuid.uuid4())
    if new_id not in choices:
        print("---add choice---")
        choices.append(new_id)
    return gr.update(value = new_id, choices=choices), choices
        

def db_add(user_id, thread_id):
    """Save user-thread id into database"""
    print("---add db---")

    from psycopg import connect
    conn = connect(PG_URI, autocommit=True)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO user_thread (user_id, thread_id)
        VALUES (%s, %s)
        ON CONFLICT (user_id, thread_id) DO NOTHING
    """, (user_id, thread_id))

def check_available_thread(user_id):
    """Retrieve existed thread for user, if its a new user, create new thread"""
    print("---check thread---")

    from psycopg import connect
    conn = connect(PG_URI, autocommit=True)
    cursor = conn.cursor()
    cursor.execute(f"""SELECT * FROM user_thread WHERE user_id = %s""", (user_id,))
    rows = cursor.fetchall()
    if rows:
        print(f"User id: {user_id} has {len(rows)} available thread")
        avai_thread = [row[1] for row in rows]
        return gr.update(value=avai_thread[0], choices=avai_thread), avai_thread
    else:
        print("New User")
        new_thread_id = str(uuid.uuid4()) 
        db_add(user_id, new_thread_id)

        return gr.update(value=new_thread_id, choices=[new_thread_id]), [new_thread_id]