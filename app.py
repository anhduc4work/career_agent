import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import ToolMessage, SystemMessage, AIMessage, HumanMessage
import uuid
from main_agent import career_agent
react_graph_memory = career_agent()

def process_file(cv_path):
    """Process link file pdf"""
    if not cv_path:
        return "No file uploaded!", gr.update(visible=True)
    
    loader = PyPDFLoader(cv_path)
    pages = loader.load_and_split()
    text = "\n".join([page.page_content for page in pages])
    return text  

def hide():
    """Hidecomponent"""
    return gr.update(visible=False)

def show():
    """Show component"""
    return gr.update(visible=True)

def check_for_new_cv(config):
    """Load new CV"""
    return react_graph_memory.get_state(config).values.get("new_cv", "Not available")

def set_environment(key, value):
    """Set environment key"""
    import os
    os.environ[key] = value
    print(f"Environment variable set: {key}")
    pass

def update_config_and_ui(thread_id, user_id):
    """Update config and its state"""
    config = {"configurable":{"thread_id": thread_id, "user_id": user_id}}
    state = react_graph_memory.get_state(config).values
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
    print("-u-")
    if user_message["files"]:
        file_content = process_file(user_message["files"][0])
        return gr.MultimodalTextbox(value=None), chat_history + [
            {"role": "user", "content": user_message["text"], "metadata": {"id": str(uuid.uuid4())}},
            {"role": "user", "content": file_content, "metadata": {"title": "File included", "id": str(uuid.uuid4())}},            
            ], file_content
    else:
        return gr.MultimodalTextbox(value=None), chat_history + [
            {"role": "user", "content": user_message["text"], "metadata": {"id": str(uuid.uuid4())}},
            ],   gr.Textbox(label="CV Content", interactive=True, visible=True)

import time
def bot(config, chat_history):
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
    bot_message = react_graph_memory.invoke(state, config)

    chat_history.append({"role": "assistant", "content": "", "metadata": {"id":bot_message["messages"][-1].id}})
    for character in bot_message["messages"][-1].content:
        chat_history[-1]['content'] += character
        time.sleep(0.001)
        yield chat_history

    chat_history[-1]['content'] = bot_message["messages"][-1].content
    return chat_history

def handle_edit(history, edit_data: gr.EditData):
    new_history = history[:edit_data.index]
    id = history[edit_data.index]["metadata"]["id"]
    new_history.append({"role": "user", "content": edit_data.value, "metadata": {"id": id}})
    return new_history

def fork_messages(config, chat_history):
    hist = react_graph_memory.get_state_history(config)
    last_message = chat_history[-1]

    for i, s in enumerate(hist):
        if i%2==1:
            pass
        else:
            if s.values["messages"][-1].id == last_message["metadata"]["id"]:
                to_fork = s.config
                break

    # update
    fork_config = react_graph_memory.update_state(to_fork,
                        {"messages": [HumanMessage(content='do you know my name', 
                               id=last_message["metadata"]["id"])]},)
    
    fork_config["configurable"]["user_id"] = config["configurable"]["user_id"]
    return fork_config

def clear_config(config):
    # clear checkpoint_id if exist:
    if config["configurable"].get("checkpoint_id", ""):
        config["configurable"].pop("checkpoint_id")
    return config


initial_user_id = "Default User"
initial_thread_id = "Default Thread"

def regenerate_id(choices):
    new_id = str(uuid.uuid4())
    choices.append(new_id)
    return gr.update(value=new_id, choices=choices), choices

def db_add(user_id, thread_id):
    from psycopg import connect
    conn = connect("dbname=postgres user=anhduc213 password=200103 host=localhost port=5432", autocommit=True)
    cursor = conn.cursor()
    cursor.execute("""INSERT INTO user_thread (user_id, thread) VALUES (%s, %s)""", (user_id, thread_id))
    pass

def gen_user(user_choices):
    """Add user list, change thread list """
    new_user_id = str(uuid.uuid4())
    new_thread_id = str(uuid.uuid4())
    user_choices.append(new_user_id)
    print(new_user_id, new_thread_id)
    # add to db
    db_add(new_user_id, new_thread_id)

    # update choice for user and thread
    return gr.update(value=new_user_id, choices=user_choices), \
            user_choices, \
            gr.update(value=new_thread_id, choices=[new_thread_id]), \
            [new_thread_id]

def gen_thread(thread_choices, user_id):
    new_thread_id = str(uuid.uuid4())
    thread_choices.append(new_thread_id)
    print(user_id, new_thread_id)
    db_add(user_id, new_thread_id)

    return gr.update(value=new_thread_id, choices=thread_choices), thread_choices

################## UI ##################

with gr.Blocks(fill_width=True) as demo:
    with gr.Row(visible=False) as row1:
        key = gr.Dropdown(
            choices=["OLLAMA_URL", "OPENAI_API_KEY"],
            label="Choose API Key",
            interactive=True,
            value= 'OPENAI_API_KEY',
            scale=1
        )
        value = gr.Textbox(label="Value", interactive=True, visible=True, placeholder="Input API key", scale=2)
        submit_url = gr.Button("Submit", scale = 0)


    gr.Markdown("# Career Agent")
    with gr.Row() as row2:  
        with gr.Column(scale = 1):   
            
            with gr.Row():
                user_choices_state = gr.State([initial_user_id])  # Gradio state to track the choices
                user_id = gr.Dropdown(
                    label="User ID",
                    value=initial_user_id,
                    choices=[initial_user_id],
                    interactive=True,
                    # scale=4,
                    allow_custom_value=True
                )
                add_user = gr.Button("+", scale=0, size="sm",)

            with gr.Row():
                thread_choices_state = gr.State([initial_thread_id])  # Gradio state to track the choices
                thread_id = gr.Dropdown(
                    label="Thread ID",
                    value=initial_thread_id,
                    choices=[initial_thread_id],
                    interactive=True,
                    # scale=4,         
                )
                add_thread = gr.Button("+", scale=0, size="sm",)



            # update = gr.Button("Update & Start")
            
            config = gr.JSON(visible=False, value = {"configurable":{"thread_id": initial_thread_id, "user_id": initial_user_id}})
            
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(type="messages", show_copy_button=True, editable="user")
            with gr.Row():  
                msg = gr.MultimodalTextbox(file_types=[".pdf"], show_label=False, placeholder="Input chat")

    with gr.Row(visible = False) as row3:
        gr.Markdown("# Underthehood")
        reload_new_cv_button = gr.Button("Reload")
        download_cv_button = gr.Button("Download")

    with gr.Row(visible = False) as row4:
        cv_text = gr.Textbox(label="CV Content", interactive=True, visible=True)
        new_cv_text = gr.Textbox(label="Reviewed CV", interactive=False, visible=False)
        # cp = gr.HighlightedText(
        #         label="Diff",
        #         combine_adjacent=True,
        #         show_legend=True,
        #         # color_map={"+": "green"},
        #         min_width=800)
            





    ############## function call ############

    # compare new cv and old one
    # new_cv_text.change(diff_texts, [cv_text, new_cv_text], [cp])

    # update drop list
    # add_thread.click(regenerate_id, [thread_choices_state], [thread_id, thread_choices_state])
    # add_user.click(regenerate_id, [user_choices_state], [user_id, user_choices_state])
    add_user.click(gen_user, [user_choices_state], [user_id, user_choices_state, thread_id, thread_choices_state])
    add_thread.click(gen_thread, [thread_choices_state, user_id], [thread_id, thread_choices_state])
    # gen new thread id
    reload_new_cv_button.click(check_for_new_cv, inputs=[config], outputs=[new_cv_text])

    # chatbot.undo(edit, chatbot, chatbot)
    chatbot.edit(handle_edit, chatbot, chatbot).then(fork_messages, [config, chatbot], [config]).then(bot, [config, chatbot], [chatbot]).then(clear_config, [config], [config])
    
    # submit message
    # msg.submit(respond, [msg, config, chatbot], [msg, chatbot, cv_text])  # func - input - output
    msg.submit(user, [msg, chatbot], [msg, chatbot, cv_text]).then(bot, [config, chatbot], [chatbot])  # func - input - output

    # update UI to thread id
    # update.click(update_config_and_ui, [thread_id, user_id], [chatbot, config, cv_text, new_cv_text])  # func - input - output

    # set environment key
    def check_available_thread(user_id):
        from psycopg import connect
        conn = connect("dbname=postgres user=anhduc213 password=200103 host=localhost port=5432", autocommit=True)
        cursor = conn.cursor()
        cursor.execute(f"""SELECT * FROM user_thread WHERE user_id = %s""", (user_id,))
        rows = cursor.fetchall()
        if rows:
            avai_thread = [row[2] for row in rows]
            return gr.update(value=avai_thread[0], choices=avai_thread), avai_thread
        else:
            new_thread_id = str(uuid.uuid4()) 
            db_add(user_id, new_thread_id)

            return gr.update(value=new_thread_id, choices=[new_thread_id]), [new_thread_id]
    # submit_url.click(set_environment, inputs = [key, value],)
    user_id.change(check_available_thread, [user_id], [thread_id, thread_choices_state]).then(update_config_and_ui, [thread_id, user_id], [chatbot, config, cv_text, new_cv_text])
    thread_id.change(update_config_and_ui, [thread_id, user_id], [chatbot, config, cv_text, new_cv_text])


demo.launch(share=True)