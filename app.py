from transformers import pipeline
import pandas as pd
import gradio as gr
import datetime
import os

label_mapping = ["B-AC", "B-LF", "I-LF" "O"] # list index value = label id {"B-AC": 0, "B-LF": 1, "I-LF": 2, "O": 3}

# define transformers pipeline
ner_pipeline = pipeline(model="mdlam/distilbert-ner-classification",
                      tokenizer="mdlam/distilbert-ner-classification")

# input example
examples = ["Here, an example: 'Exponential weighted moving average.'"]

# Log file path
log_file_path = "ner_log.txt"

# define interaction log array
interaction_log = []

def ner_predict(text):
    output = ner_pipeline(text)
    # Format the output for HighlightedText
    highlighted_text = [
        (text[ent['start']:ent['end']], ent['entity']) for ent in output
    ]
    # log_input = {
    #         "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    #         "text": text,
    #         "entities": output
    #     }
    log_input = "hi"
    interaction_log.append(
        f"Input: {text}\nOutput: {output}\n\n"
    )

    return highlighted_text

def create_log():
    if not os.path.exists(log_file_path):
        with open(log_file_path, "w") as file:
            file.write(interaction_log)

    return log_file_path

with gr.Blocks() as demo:
    with gr.Row():
      with gr.Column():
        input_box = gr.Textbox(label="Input", placeholder="Enter sentence here...")
        submit_button = gr.Button("Submit")
        gr.Examples(examples, input_box)
      with gr.Column():
          output_box = gr.HighlightedText(label="Entities")

          log_button = gr.Button("Generate Log")
          log_file_output = gr.File(label="Dowload Log")

    submit_button.click(fn=ner_predict, inputs=input_box, outputs=output_box)

    log_button.click(fn=create_log, inputs=None, outputs=log_file_output)

demo.launch()