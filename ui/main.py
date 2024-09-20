import time
import gradio as gr
import weaviate, os

#############################################################
#
#  Bibliography:
#    Gradio theme https://huggingface.co/spaces/gradio/theme-gallery
#
#    Gradio Tutorial:
#         https://www.youtube.com/watch?v=ABNxNFPqIGQ&t=4s
#         https://www.youtube.com/watch?v=44vi31hehw4
#
#   DeepLearning.AI (https://learn.deeplearning.ai/courses/building-multimodal-search-and-rag/lesson/1/introduction)
#   platform course: Building Multimodal Search and RAG
###############################################################

import os

from click import prompt
from dotenv import load_dotenv, find_dotenv
import textwrap
import PIL.Image
import google.generativeai as genai
from google.api_core.client_options import ClientOptions

_ = load_dotenv(find_dotenv('../config/.env')) # read local .env file
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

print(f"The Gemmi API KEY found is {GOOGLE_API_KEY}")
#Configure API KEY GEMMINI
genai.configure(api_key=GOOGLE_API_KEY)


SYSTEM_PROMPT = ""

SYSTEM_PROMPT_IDENTIFIED = "Explain what you see in this image."

def to_markdown(text):
    text = text.replace('•', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)


def call_Gemmi_LLM(image_path: str, prompt: str) -> str:
    # Load the image
    # img = PIL.Image.open(image_path)

    sample_file = genai.upload_file(path=image_path, display_name="Sample drawing")

    #
    # Call generative model
    #
    #

    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

    response = model.generate_content(
        [prompt, sample_file]
    )

    return to_markdown(response.text)

def inference(prompt, inp_img, inp_video, token):
    time.sleep(3)
    result_call_video_gemmini = result_call_image_gemmini = ''
    prompt = SYSTEM_PROMPT_IDENTIFIED

    if inp_img != '' and inp_img is not None:
      result_call_image_gemmini = call_Gemmi_LLM(inp_img, prompt)

    if inp_video != '' and inp_video is not None:
        print(f"Video Address to show and index ${inp_video}")
        result_call_video_gemmini = call_Gemmi_LLM(inp_video, prompt)

    return [f" Prompt {prompt} \n\n Image:\n {result_call_image_gemmini},\n\n\n Video:\n {result_call_video_gemmini}","B"]

def multimodalraginference(*args):
    gr.Warning("Building action!!!!")
    return

def multimodalrecomendation(*args):
    gr.Warning("Building action!!!!")
    return

EMBEDDING_API_KEY = os.getenv("WEVIATE_API_ADMIN_KEY")
# client = weaviate.connect_to_embedded(
#     version="1.24.21",
#     environment_variables={
#         "ENABLE_MODULES": "backup-filesystem,multi2vec-palm",
#         "BACKUP_FILESYSTEM_PATH": "../backups",
#     },
#     headers={
#         "X-PALM-Api-Key": EMBEDDING_API_KEY,
#     }
# )
#
# client.is_ready()

#########################################
#
#   Bibliografy:
#     Google AI Generative AI with Weaviate (https://weaviate.io/developers/weaviate/model-providers/google/generative)
#
#
###########################################################

with gr.Blocks(theme = 'JohnSmith9982/small_and_pretty') as demo:
    with gr.Tab("Multimodal Search "):
        message = ("### Use LLM Multimodal Prompt search with Gradio using these \
        [video tutorial](https://www.youtube.com/watch?v=ABNxNFPqIGQ&t=4s) and Hugging Face API")
        gr.Markdown("<center><h2>Open LLM Explorer</h2></center>")
        gr.Markdown(message)

        prompt = gr.Textbox(label = 'Prompt', lines= 3, max_lines = 5, value = SYSTEM_PROMPT_IDENTIFIED)
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
                generate_btn = gr.Button("Generate", size = "lg", variant = "primary")
                code_btn = gr.Button("View Code", size = "lg", variant = "secondary")

        with gr.Row() as row_output:
           llama_output = gr.Markdown("##Llama3.1 70B-instruct Output")
           groq_output = gr.Markdown("##Groq with Llama3.1 70B-instruct Output")

        generate_btn.click(fn= inference, inputs=[prompt, inp_img, inp_video, token],outputs=[llama_output,groq_output])
    with gr.Tab("RAG Multimodal"):
        ragMultimodal_button = gr.Button("In Building")

        ragMultimodal_button.click(fn = multimodalraginference)
    with gr.Tab("Multimodal Recommendation"):
        ragMultimodal_button = gr.Button("In Building")

        ragMultimodal_button.click(fn = multimodalrecomendation)


demo.launch()
