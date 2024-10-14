from transformers import BertTokenizer, BertModel

# Check if BERT model loads correctly
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    print("BERT is installed and loaded successfully!")
except Exception as e:
    print(f"An error occurred: {e}")
