from transformers import pipeline
import pandas as pd
import gradio as gr
import datetime
import os

label_mapping = {"LABEL_0": "B-AC", "LABEL_1": "B-LF", "LABEL_2": "I-LF", "LABEL_3": "O"}

# define transformers pipeline
ner_pipeline = pipeline(model="mdlam/distilbert-ner-classification",
                      tokenizer="mdlam/distilbert-ner-classification")

# input example
examples = ["Here, an example: 'Exponential weighted moving average.'", "I don't understand the historical significance of the Roman Empire.", "Preliminary results show that a tax levied on sugar-sweetened beverages (SSBs) by the Portuguese government in 2017 led to a drop in sales and reformulation of these products.", "The UN and the EU is researching the documentation of the differentiation of regions.", "In most patients, preoperative levels of serum follicle-stimulating hormone (FSH), luteinizing hormone (LH), and total testosterone were measured."]

# Log file path
log_file_path = "ner_log.txt"

# define interaction log array
interaction_log = []

def ner_predict(text):
    output = ner_pipeline(text)
    # Format the output for HighlightedText
    highlighted_text = [
        (text[ent['start']:ent['end']], label_mapping.get(ent['entity'], ent['entity']))
        for ent in output
    ]


    # log_input = "hi"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    interaction_log.append(
        f"[{timestamp}]\nInput: {text}\nOutput: {highlighted_text}\n\n"
    )

    return highlighted_text

def create_log():
    if not os.path.exists(log_file_path):
        with open(log_file_path, "w") as file:
            file.writelines(interaction_log)
    else:
         os.remove("ner_log.txt")
         with open(log_file_path, "w") as file:
            file.writelines(interaction_log)


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