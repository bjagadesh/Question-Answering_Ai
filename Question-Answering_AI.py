from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import os
from nltk import word_tokenize

# Load pre-trained question answering model and tokenizers
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

def extract_answer(question, text):
    # Tokenize input question and text
    inputs = tokenizer(question, text,return_tensors="pt")
    #print(inputs,"inputs")
    


    # Generate predictions
    with torch.no_grad():
        outputs = model(**inputs)
        
    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    answer=tokenizer.decode(predict_answer_tokens)

    return answer


# Read the text file
file_path = "F:\jagad\python\stockmarket\Regularincome.txt"  # Replace with the actual file path
with open(file_path, 'r') as file:
    content = file.read()

# Process question on the text file
question = input("Enter a question:")
#print("Question:", question)

# Split content into smaller chunks
tokens = word_tokenize(content)
#print(tokens)
max_length = 400  # Maximum token limit for the model

chunks = []
i = 0
while i < len(tokens):
    chunk_tokens = tokens[i:i+max_length]
    if len(chunk_tokens) > 0:
        chunk = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk)
        i += len(chunk_tokens)
    else:
        i += 1

# Process each chunk and extract the answer
for chunk in chunks:
    #print(len(chunk),"len(chunk)")
    #print(chunk)
    answer = extract_answer(question, chunk)
    if "[CLS]" in answer:
        continue
    else:
        print("Answer:", answer)
        break


