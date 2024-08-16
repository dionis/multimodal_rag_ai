import time
import gradio as gr

#############################################################
#
#  Bibliography:
#    Gradio theme https://huggingface.co/spaces/gradio/theme-gallery
#
#    Gradio Tutorial:
#         https://www.youtube.com/watch?v=ABNxNFPqIGQ&t=4s
#         https://www.youtube.com/watch?v=44vi31hehw4
#
###############################################################

SYSTEM_PROMPT = ""

def inference(*args):
    time.sleep(1)
    return ["A","B"]
with gr.Blocks(theme='JohnSmith9982/small_and_pretty') as demo:
    message = ("### Use LLM Multimodal Prompt search with Gradio using these \
    [video tutorial](https://www.youtube.com/watch?v=ABNxNFPqIGQ&t=4s) and Hugging Face API")
    gr.Markdown("<center><h2>Open LLM Explorer</h2></center>")
    gr.Markdown(message)

    prompt = gr.Textbox(label='Prompt', lines= 3, max_lines = 5)
    with gr.Row():
        # Image gradio component bibliography https://www.gradio.app/docs/gradio/image
        #
        inp_img = gr.Image(label="Image to Search",type="filepath")

        # Video gradio component bibliography https://www.gradio.app/docs/gradio/video
        #
        #
        inp_video = gr.Video(label="Video to Search")
    token = gr.Textbox(label='Token', type='password')


    with gr.Group():
        with gr.Row():
            generate_btn = gr.Button("Generate", size="small", variant="primary")
            code_btn = gr.Button("View Code", size="small", variant="secondary")

    with gr.Row() as row_output:
       llama_output = gr.Markdown("##Llama3.1 70B-instruct Output")
       groq_output = gr.Markdown("##Groq with Llama3.1 70B-instruct Output")

    generate_btn.click(fn= inference, inputs=[prompt, inp_img, inp_video, token],outputs=[llama_output,groq_output])

demo.launch()
