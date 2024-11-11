import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle


class TextPredictor:
    def __init__(self, model_path='best_model.pth',
                 tokenizer_path='tokenizer/',
                 label_mapping_path='label_mapping.pkl'):
        # Load label mapping
        with open(label_mapping_path, 'rb') as f:
            mappings = pickle.load(f)
            self.inverse_label_mapping = mappings['inverse_label_mapping']

        # Initialize tokenizer and model
        self.model_name = "ai4bharat/indic-bert"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.inverse_label_mapping)
        )

        # Load trained model weights
        self.model.load_state_dict(torch.load(model_path))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        # Tokenize input
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Move tensors to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

        # Get predicted label and probability
        predicted_label = self.inverse_label_mapping[predicted_class]
        confidence = probabilities[0][predicted_class].item()

        return {
            'label': predicted_label,
            'confidence': confidence
        }


def main():
    predictor = TextPredictor()

    texts = ["അവർ മരിക്കുമെന്ന് ഞാൻ പ്രതീക്ഷിക്കുന്നു","Eathu nayinte makkal dislike cheythu","Super casting",
             "Dislike adichavar ettavum valiya oolakal...... bldy idiots...", "Dislike adikan vendi mathram trailer kanunna kure vanangalu",
             "Al veruppikkal Mammotty ude vere oru adipoli veruppikkal padamayirikku ith", "മൈരൻ മോഹനന്റെ ഒടിയൻ ഫാൻസ് മൊത്തം കുരു വാരി എറിയുന്നുണ്ടല്ലോ",
             "Eathu thendigal dislike cheydhu","Trailer kandal ariyam nalla oombiya padam aanunn"]
    
    for text in texts:
        result = predictor.predict(text)

        print(text)
        print(f"Predicted Label: {result['label']}",end=' ')
        if result['label'] == 0:
            print("Not offensive")
        elif result['label'] == 1:
            print("Offensive")
        elif result['label'] == 2:
            print("Offensive, Targeting an Individual")
        else:
            print("Offensive, Targeting a Group")
        print(f"Confidence: {result['confidence']:.4f}")
        print()


if __name__ == "__main__":
    main()