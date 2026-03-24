import torch
from torch.utils.data import Dataset
import pandas as pd

class ClarityEvasionDataset(Dataset):
    """
    Dataset PyTorch personalizzato per il task di Evasion Classification (Task 2).
    Prende i dati di HuggingFace, struttura il testo e lo tokenizza.
    """
    def __init__(self, hf_dataset, tokenizer, max_len, evasion2id):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.evasion2id = evasion2id
        
        # Recuperiamo il token separatore corretto (es. [SEP] per BERT, </s> per Longformer)
        self.sep_token = tokenizer.sep_token if tokenizer.sep_token else " "

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Estraiamo le feature dalla riga corrente
        item = self.data[index]
        
        # Gestiamo eventuali valori nulli
        question = str(item.get('question', ''))
        context = str(item.get('interview_question', ''))
        answer = str(item.get('interview_answer', ''))
        
        # ---------------------------------------------------------
        # PROMPT ENGINEERING PER ENCODER (Cruciale per i risultati)
        # ---------------------------------------------------------
        # Formato: Domanda Specifica [SEP] Contesto (Domanda Lunga) [SEP] Risposta
        structured_text = f"Question: {question} {self.sep_token} Context: {context} {self.sep_token} Answer: {answer}"
        
        # Tokenizzazione
        encoding = self.tokenizer(
            structured_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # ---------------------------------------------------------
        # GESTIONE DELLE ETICHETTE MANCANTI (Per i Test Set ciechi)
        # ---------------------------------------------------------
        # Prendi l'etichetta. Se non esiste (None), mettila vuota
        label_str = item.get('evasion_label', '')
        
        # Se l'etichetta è una stringa valida tra le nostre 9 classi
        if label_str and label_str in self.evasion2id:
            label_id = self.evasion2id[label_str]
        else:
            # Se siamo sul Test set ufficiale (che non ha le vere label)
            # mettiamo un valore dummy (-1). La CrossEntropy del Trainer le ignorerà,
            # ma noi vogliamo solo le predict. Se usiamo 0 (che è Explicit), almeno non crasha.
            # Impostiamo -1, che il Trainer HuggingFace di default ignora.
            label_id = -100

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_id, dtype=torch.long)
        }