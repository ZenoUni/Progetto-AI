import os
import random
import subprocess
import yaml
import zipfile
import numpy as np
import torch

# Definisci il numero di classi e i nomi delle classi direttamente nel codice
NUM_CLASSES = 5
CLASS_NAMES = ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']


# Funzione per impostare il seed per la riproducibilità
def set_seed(seed=None):
    if seed is None:
        seed = random.randint(0, 2 ** 32 - 1)  # Genera un seed casuale
        print(f"Seed casuale generato: {seed}")
    else:
        print(f"Seed predefinito utilizzato: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


# Carica il file di configurazione YAML
def load_config(config_path='config.yaml', required_keys=None):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Il file di configurazione {config_path} non è stato trovato.")
    except yaml.YAMLError as e:
        raise ValueError(f"Errore nel parsing del file di configurazione YAML: {e}")

    # Validazione delle chiavi richieste
    if required_keys:
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise KeyError(f"Chiavi mancanti nel file di configurazione: {missing_keys}")

    return config


# Funzione per eseguire comandi di sistema con gestione degli errori
def run_command(command, description="comando"):
    try:
        print(f"Esecuzione di: {' '.join(command)}")
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Errore durante l'esecuzione del {description}: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Comando non trovato: {command[0]}")


# Funzione per scaricare il dataset personalizzato
def download_personal_dataset():
    personal_dir = os.path.join("yolov5", "Dataset", "Personal")
    if not os.path.exists(personal_dir):
        print("Creazione della cartella 'Personal' e download del dataset personalizzato...")
        os.makedirs(personal_dir, exist_ok=True)
        repo_url = "https://github.com/ZenoUni/AI-MachineLearning"

        # Clona il repository (contiene solo le cartelle images e labels)
        subprocess.run(["git", "clone", "--depth", "1", repo_url, personal_dir], check=True)

        print("Dataset personalizzato scaricato e preparato in 'yolov5/Dataset/Personal'.")
    else:
        print("Cartella 'Personal' già presente. Skip del download.")


# Funzione per scaricare e scompattare il dataset
def download_and_unzip_dataset():
    DATASET_URL = "https://public.roboflow.com/ds/xKLV14HbTF?key=aJzo7msVta"
    DATASET_DIR = os.path.join("yolov5", "Dataset")

    if not os.path.exists(DATASET_DIR):
        print("Creazione della cartella 'Dataset'...")
        os.makedirs(DATASET_DIR, exist_ok=True)

    if not os.path.exists(os.path.join(DATASET_DIR, 'train')):  # Controlla se il dataset è già presente
        print("Scaricamento del dataset...")
        subprocess.run(["curl", "-L", DATASET_URL, "-o", "roboflow.zip"], check=True)

        # Estrazione del file ZIP utilizzando zipfile (compatibile con Windows)
        try:
            with zipfile.ZipFile("roboflow.zip", 'r') as zip_ref:
                print("Scompattamento del dataset...")
                zip_ref.extractall(DATASET_DIR)
            os.remove("roboflow.zip")
        except zipfile.BadZipFile:
            raise RuntimeError("Errore durante l'estrazione del file ZIP. File corrotto o non valido.")

        print("Dataset scaricato e scompattato con successo.")
    else:
        print("Dataset già presente. Skip del download.")


# Clona il repository YOLOv5 se non è già stato scaricato
def clone_yolov5(repo_url="https://github.com/ultralytics/yolov5", dest_dir="yolov5"):
    if not os.path.exists(dest_dir):
        print("Clonazione del repository YOLOv5...")
        run_command(["git", "clone", repo_url, dest_dir], description="clonazione del repository YOLOv5")
    else:
        print("Repository YOLOv5 già presente.")


# Installa i requisiti nel caso non siano già installati
def install_requirements(requirements_path="yolov5/requirements.txt"):
    print("Installazione dei pacchetti richiesti...")
    subprocess.run(["pip", "install", "-r", requirements_path], check=True)


# Prepara il file di configurazione dei dati YOLOv5
def prepare_data_yaml(config, yaml_path="data.yaml"):
    data_config = {
        'train': config['train_data_path'],
        'val': config['val_data_path'],
        'nc': NUM_CLASSES,  # Usa NUM_CLASSES direttamente dallo script
        'names': CLASS_NAMES  # Usa CLASS_NAMES direttamente dallo script
    }
    try:
        with open(yaml_path, 'w') as file:
            yaml.dump(data_config, file)
        print(f"File di configurazione dati salvato come {yaml_path}.")
    except IOError as e:
        raise RuntimeError(f"Errore durante la scrittura del file YAML {yaml_path}: {e}")


# Allena il modello YOLOv5 con il dataset personalizzato
def train_model(yaml_path="data.yaml", epochs=5, img_size=128, batch_size=16):
    print("Avvio dell'addestramento del modello...")
    subprocess.run([
        "python", "yolov5/train.py",
        "--img", str(img_size),
        "--batch", str(batch_size),
        "--epochs", str(epochs),
        "--data", yaml_path,
        "--weights", "yolov5s.pt",
        "--project", "yolov5/runs/train",  # Directory di log per TensorBoard
        "--name", "exp"  # Nome dell'esperimento
    ], check=True)


# Funzione per ottenere l'ultimo esperimento (expX) eseguito
def get_latest_exp_folder(base_dir="yolov5/runs/train"):
    exp_folders = [f for f in os.listdir(base_dir) if f.startswith("exp") and (f[3:].isdigit() or f == "exp")]

    if not exp_folders:
        raise ValueError("Non ci sono cartelle di esperimenti validi (expX).")

    latest_exp = max(exp_folders, key=lambda f: int(f[3:]) if f != 'exp' else 0)
    return latest_exp


# Rileva oggetti in un dataset personalizzato utilizzando il modello addestrato
def detect_objects(source_path, weights_path=None):
    if weights_path is None:
        weights_path = get_best_weight_from_all_exps()

    print(f"Avvio della rilevazione degli oggetti usando il peso: {weights_path}")

    latest_exp = get_latest_exp_folder()
    output_dir = os.path.join("yolov5/runs/detect", latest_exp)
    os.makedirs(output_dir, exist_ok=True)

    # Verifica che la cartella di test esista
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"{source_path} does not exist")

    # Esegui il comando di rilevazione
    command = [
        "python", "yolov5/detect.py",
        "--weights", weights_path,
        "--source", source_path,  # Aggiorna con il percorso corretto
        "--img", "640",
        "--conf", "0.25",
        "--project", output_dir,
        "--name", "results"
    ]

    subprocess.run(command, check=True)


# Funzione per ottenere il miglior peso tra tutti gli esperimenti
def get_best_weight_from_all_exps(base_dir="yolov5/runs/train"):
    exp_folders = [f for f in os.listdir(base_dir) if f.startswith("exp")]

    best_weight = None
    latest_time = 0

    for exp in exp_folders:
        weight_path = os.path.join(base_dir, exp, "weights", "best.pt")
        if os.path.exists(weight_path):
            mod_time = os.path.getmtime(weight_path)
            if mod_time > latest_time:
                latest_time = mod_time
                best_weight = weight_path

    if best_weight:
        return best_weight
    else:
        raise FileNotFoundError("Non è stato trovato alcun peso 'best.pt' negli esperimenti.")


# Avvio di TensorBoard per la visualizzazione
def start_tensorboard(log_dir="yolov5/runs/train"):
    print("Avvio di TensorBoard...")
    subprocess.run(["tensorboard", "--logdir", log_dir, "--bind_all"], check=True)


# Esegui lo script principale
if __name__ == "__main__":
    # Specifica le chiavi richieste per il file di configurazione
    required_keys = ["use_fixed_seed", "train_data_path", "val_data_path", "epochs", "img_size", "batch_size"]

    # Carica e valida il file di configurazione
    config = load_config("config.yaml", required_keys=required_keys)

    # Imposta il seed per la riproducibilità (fisso o casuale)
    use_fixed_seed = config.get("use_fixed_seed", True)
    seed = config.get("seed") if use_fixed_seed else None
    actual_seed = set_seed(seed)

    print(f"Seed attivo: {actual_seed}")

    # Step 1: Clona YOLOv5
    clone_yolov5()

    # Step 2: Scarica e scompattare il dataset da Roboflow
    download_and_unzip_dataset()

    # Step 3: Scarica il dataset personalizzato
    download_personal_dataset()

    # Step 4: Installa i requirements
    install_requirements()

    # Step 5: Prepara il file YAML per il dataset
    prepare_data_yaml(config)

    # Step 6: Addestra il modello
    train_model(
        epochs=config.get("epochs", 5),
        img_size=config.get("img_size", 128),
        batch_size=config.get("batch_size", 16)
    )

    # Step 7: Avvia TensorBoard per monitorare l'addestramento
    start_tensorboard()

    # Step 8: Rileva oggetti nel dataset di test
    test_images_path = os.path.join("yolov5", "Dataset", "test", "images")
    detect_objects(test_images_path)