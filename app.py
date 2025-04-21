import gradio as gr
from app_function import (check_available_thread, 
                          update_config_and_ui, 
                          update_dropdown,
                          db_add,
                          handle_edit,
                          fork_messages,
                          bot,
                          user,
                          clear_checkpoint_in_config,
                          update_underthehood_ui)


################## UI ##################
initial_user_id = "Default User"
initial_thread_id = "1"
db_add(initial_user_id, initial_thread_id)

with gr.Blocks(fill_width=True) as demo:
    
    gr.Markdown("# \n\n\n ___",height=40)
    gr.Markdown("# Career Agent")
    with gr.Tab("Chat"):
        with gr.Row() as row2:  
            with gr.Column(scale = 1):   
                
                with gr.Row():
                    user_choices_state = gr.State([initial_user_id])  # Gradio state to track the choices
                    user_id = gr.Dropdown(
                        value=initial_user_id, choices=[initial_user_id],
                        label="User ID", interactive=True, allow_custom_value=True
                    )
                    add_user = gr.Button("+", scale=0, size="sm",)

                with gr.Row():
                    thread_choices_state = gr.State([initial_thread_id])  # Gradio state to track the choices
                    thread_id = gr.Dropdown(
                        value=initial_thread_id, choices=[initial_thread_id],
                        label="Thread ID", interactive=True,
                    )
                    add_thread = gr.Button("+", scale=0, size="sm")
                
                config = gr.JSON(visible=False, value = {"configurable":{"thread_id": initial_thread_id, "user_id": initial_user_id}})
                
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(type="messages", show_copy_button=True, editable="user")
                msg = gr.MultimodalTextbox(file_types=[".pdf"], show_label=False, placeholder="Input chat")

    with gr.Tab("Underthehood") as tab2:

        with gr.Column():
            cross_thread_info = gr.Textbox(label="User Info (Cross Thread)", interactive=False, visible=True)
            single_thread_summary = gr.Textbox(label="Thread Summary", interactive=False, visible=True)
        
        with gr.Row():
            cv_text = gr.Textbox(label="CV Content", interactive=True, visible=True)
            new_cv_text = gr.Textbox(label="Reviewed CV", interactive=False, visible=True)


        cp = gr.HighlightedText(
                label="Diff",
                combine_adjacent=True,
                show_legend=True,
                # color_map={"+": "green"},
                min_width=800)
            


    ############## function call ############

    # auto load existed data when loading app
    demo.load(check_available_thread, [user_id], [thread_id, thread_choices_state]).\
        then(update_config_and_ui, [thread_id, user_id], [chatbot, config, cv_text, new_cv_text])

    # add new id
    add_user.click(update_dropdown, [user_choices_state], [user_id, user_choices_state]).\
        then(update_dropdown, [thread_choices_state], [thread_id, thread_choices_state]).\
        then(db_add, [user_id, thread_id]).\
        then(update_config_and_ui, [thread_id, user_id], [chatbot, config, cv_text, new_cv_text])
    add_thread.click(update_dropdown, [thread_choices_state], [thread_id, thread_choices_state]).\
        then(db_add, [user_id, thread_id]).\
        then(update_config_and_ui, [thread_id, user_id], [chatbot, config, cv_text, new_cv_text])
        
    # if user is updated
    user_id.input(check_available_thread, [user_id], [thread_id, thread_choices_state]).\
        then(update_dropdown, [user_choices_state, user_id], [user_id, user_choices_state]).\
        then(update_config_and_ui, [thread_id, user_id], [chatbot, config, cv_text, new_cv_text])
    
    #if user want to change id
    thread_id.select(update_config_and_ui, [thread_id, user_id], [chatbot, config, cv_text, new_cv_text])

    # fork messages
    chatbot.edit(handle_edit, chatbot, chatbot).\
        then(fork_messages, [config, chatbot], [config]).\
        then(bot, [config, chatbot], [chatbot]).\
        then(clear_checkpoint_in_config, [config], [config])
    
    # submit message
    msg.submit(user, [msg, chatbot], [msg, chatbot]).\
        then(bot, [config, chatbot], [chatbot])  # 
        
    # show backend
    tab2.select(update_underthehood_ui, [config], [new_cv_text, single_thread_summary, cross_thread_info])
   
    # compare new cv and old one
    # new_cv_text.change(diff_texts, [cv_text, new_cv_text], [cp])
    
    
demo.launch(share=True)