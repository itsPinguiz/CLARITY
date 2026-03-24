import os

# ==========================================
# 1. MODALITÀ ESECUZIONE (PC Locale vs Colab)
# ==========================================
# Imposta a True quando scrivi codice sul tuo PC (GTX 1060).
# Imposta a False prima di lanciare il training lungo su Colab.
DEBUG_MODE = False

# ==========================================
# 2. PATHS E DIRECTORY
# ==========================================
# Otteniamo la cartella root in cui ci troviamo (che sia locale o Colab)
ROOT_DIR = os.getcwd()

# Creiamo le cartelle se non esistono (ignorate da git tramite .gitignore)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ==========================================
# 3. IMPOSTAZIONI MODELLO E DATI
# ==========================================
DATASET_URL = "ailsntua/QEvasion"

if DEBUG_MODE:
    # Setup ultraleggero per testare il codice in locale senza fondere la GPU
    MODEL_NAME = "prajjwal1/bert-tiny" 
    MAX_LEN = 128
    BATCH_SIZE = 2
    EPOCHS = 1
    SAMPLE_SIZE = 50 # Prendiamo solo 50 esempi per fare debug veloce
else:
    # Setup da Gara/Ricerca per Google Colab (GPU T4 15GB)
    MODEL_NAME = "allenai/longformer-base-4096"
    MAX_LEN = 1024 # Copre il 95%+ del dataset senza OOM sulla T4
    BATCH_SIZE = 4 # Da abbinare a gradient_accumulation_steps=4 o 8
    EPOCHS = 5
    SAMPLE_SIZE = None # Usa tutto il dataset

LEARNING_RATE = 2e-5

# ==========================================
# 4. LABEL MAPPING (Task 1 e Task 2)
# ==========================================
# TASK 2: 9 Tecniche di Evasione (Il nostro target di training)
EVASION_LABELS = [
    "Explicit", "Implicit", "Dodging", "General", "Deflection",
    "Partial/half-answer", "Declining to answer", "Claims ignorance", "Clarification"
]
# Dizionari per convertire stringhe <-> interi per PyTorch
EVASION2ID = {label: i for i, label in enumerate(EVASION_LABELS)}
ID2EVASION = {i: label for i, label in enumerate(EVASION_LABELS)}

# TASK 1: 3 Classi di Chiarezza Macro
CLARITY_LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
CLARITY2ID = {label: i for i, label in enumerate(CLARITY_LABELS)}
ID2CLARITY = {i: label for i, label in enumerate(CLARITY_LABELS)}

# ==========================================
# 5. GERARCHIA: MAPPATURA TASK 2 -> TASK 1
# ==========================================
# Questa è la nostra "arma segreta". Alleniamo sulle 9 classi, 
# e deduciamo le 3 macro-classi deterministicamente per avere coerenza perfetta al 100%.
EVASION_TO_CLARITY_MAP = {
    "Explicit": "Clear Reply",
    
    "Implicit": "Ambivalent",
    "General": "Ambivalent",
    "Partial/half-answer": "Ambivalent",
    "Dodging": "Ambivalent",
    "Deflection": "Ambivalent",
    
    "Declining to answer": "Clear Non-Reply",
    "Claims ignorance": "Clear Non-Reply",
    "Clarification": "Clear Non-Reply"
}