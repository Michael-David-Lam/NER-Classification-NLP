---
title: NER Classification NLP
emoji: 😻
colorFrom: yellow
colorTo: indigo
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
short_description: App demo of NLP NER classification model
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
# NER Classification App 

Hosted on HuggingFace Spaces, this project showcases a Named Entity Recognition (NER) model built using DistilBERT and deployed through a Gradio web interface. The app takes user-input text, predicts entity labels, and highlights them in the UI. 
The model was determined through comparitive experimentation setups and trained on the [PLOD-CW-25](https://huggingface.co/datasets/surrey-nlp/PLOD-CW-25) dataset. Feel free to test it out with various phrases and words to see how it classifies each label.

[NER Classification App](https://huggingface.co/spaces/mdlam/NER-Classification-NLP)

CLass Labels:
- B-AC: beginning of an abbreviation
- B-LF: beginning of long form
- I-LF: inside of a long form
- O: outside any named entity

