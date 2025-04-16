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

def update_config_and_ui(thread_id):
    """Update config and its state"""
    config = {"configurable":{"thread_id": thread_id}}
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



def respond(msg, config, chat_history):
    # read input file (cv)
    if msg["files"]:
        cv = process_file(msg["files"][0])
        state = {
            "messages": [HumanMessage(msg["text"])],
            "cv": cv,
        }
    else:
        state = {
            "messages": [HumanMessage(msg["text"])],
        }

    print(state, config)
    bot_message = react_graph_memory.invoke(state, config)
    
    chat_history.append({"role": "user", "content": msg["text"]})
    chat_history.append({"role": "assistant", "content": bot_message["messages"][-1].content})

    if msg["files"]:
        return gr.MultimodalTextbox(value=None), chat_history, cv
    else:
        return gr.MultimodalTextbox(value=None), chat_history, gr.Textbox(label="CV Content", interactive=True, visible=True)

# from difflib import Differ
# def diff_texts(text1, text2):
#     d = Differ()
#     return [
#         (token[2:], token[0] if token[0] != " " else None)
#         for token in d.compare(text1, text2)
#     ]

initial_id = str(uuid.uuid4())

def regenerate_thread_id(choices):
    new_id = str(uuid.uuid4())
    choices.append(new_id)
    return gr.update(value=new_id, choices=choices), choices

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
            choices_state = gr.State([initial_id])  # Gradio state to track the choices
            thread_id = gr.Dropdown(
                label="Thread ID",
                value=initial_id,
                choices=[initial_id],
                interactive=True,
                allow_custom_value=True,
                scale=4,         
            )
            with gr.Row():
                renew = gr.Button("New Thread", scale=0, size="sm")
                update = gr.Button("Update & Start")
            
            config = gr.JSON(visible=False, value = {"configurable":{"thread_id": initial_id}})
            
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(type="messages",)
            with gr.Row():  
                msg = gr.MultimodalTextbox(file_types=[".pdf"], show_label=False, placeholder="Input chat")

    with gr.Row() as row3:
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
    renew.click(regenerate_thread_id, [choices_state], [thread_id, choices_state])

    # gen new thread id
    reload_new_cv_button.click(check_for_new_cv, inputs=[config], outputs=[new_cv_text])

    # submit message
    msg.submit(respond, [msg, config, chatbot], [msg, chatbot, cv_text])  # func - input - output

    # update UI to thread id
    update.click(update_config_and_ui, [thread_id], [chatbot, config, cv_text, new_cv_text])  # func - input - output

    # set environment key
    # submit_url.click(set_environment, inputs = [key, value],)


demo.launch(share=True)