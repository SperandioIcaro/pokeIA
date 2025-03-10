import tensorflow as tf
import numpy as np
import os
import sys
from PIL import Image
from colorama import Fore, Style, init

# Configurações
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'pokemon_model.h5')
TEST_DIR = os.path.join(BASE_DIR, 'test_images')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
IMG_SIZE = 160
MIN_CONFIDENCE = 0.65  # Nova confiança mínima

def initialize():
    init(autoreset=True)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"{Fore.RED}ERRO: Modelo não encontrado em {MODEL_PATH}")
        print(f"Execute primeiro o treinamento:{Style.RESET_ALL}\n  python src/main.py")
        sys.exit(1)
    return tf.keras.models.load_model(MODEL_PATH)

def get_class_names():
    if not os.path.exists(DATASET_DIR):
        print(f"{Fore.RED}ERRO: Dataset não encontrado em {DATASET_DIR}{Style.RESET_ALL}")
        sys.exit(1)
    return sorted(os.listdir(DATASET_DIR))

def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        # Aumento de dados durante a predição (TTA)
        img = img.resize((IMG_SIZE + 20, IMG_SIZE + 20))
        img = tf.image.random_crop(np.array(img), size=[IMG_SIZE, IMG_SIZE, 3])
        img_array = tf.keras.applications.efficientnet.preprocess_input(img)
        return np.expand_dims(img_array, axis=0), None
    except Exception as e:
        return None, f"Erro: {str(e)}"

def postprocess_prediction(predictions, class_names, top_n=3):
    top_indices = np.argsort(predictions[0])[-top_n:][::-1]
    return [(class_names[i], predictions[0][i] * 100) for i in top_indices]

def predict_images():
    if not os.path.exists(TEST_DIR):
        print(f"{Fore.RED}ERRO: Pasta de testes não encontrada em {TEST_DIR}{Style.RESET_ALL}")
        sys.exit(1)

    model = load_model()
    class_names = get_class_names()
    
    print(f"\n{Fore.CYAN}=== SISTEMA DE RECONHECIMENTO POKÉMON ===")
    print(f"{Fore.YELLOW}• Modelo: {os.path.basename(MODEL_PATH)}")
    print(f"• Classes: {len(class_names)} Pokémon")
    print(f"• Confiança Mínima: {MIN_CONFIDENCE*100:.0f}%")
    print(f"• Imagens para Teste: {len(os.listdir(TEST_DIR))}{Style.RESET_ALL}\n")

    for filename in os.listdir(TEST_DIR):
        image_path = os.path.join(TEST_DIR, filename)
        img_array, error = preprocess_image(image_path)
        
        if error:
            print(f"{Fore.RED}✖ {filename}: {error}{Style.RESET_ALL}")
            continue

        try:
            predictions = model.predict(img_array, verbose=0)
            confidence = np.max(predictions) * 100
            
            if confidence < MIN_CONFIDENCE * 100:
                print(f"{Fore.YELLOW}⚠️  {filename}:")
                print(f"   Confiança Insuficiente: {confidence:.2f}%")
                print(f"   Sugestão: Verificar manualmente{Style.RESET_ALL}")
                continue

            top_predictions = postprocess_prediction(predictions, class_names)
            
            print(f"{Fore.GREEN}▶ {filename}:")
            for i, (poke, conf) in enumerate(top_predictions):
                prefix = "🏆" if i == 0 else f"{i+1}."
                print(f"   {prefix} {poke}: {conf:.2f}%")
            print("-" * 50)
            
        except Exception as e:
            print(f"{Fore.RED}Erro na predição: {filename} - {str(e)}{Style.RESET_ALL}")

if __name__ == "__main__":
    initialize()
    try:
        predict_images()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Operação interrompida pelo usuário{Style.RESET_ALL}")