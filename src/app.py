import os
import sys
import base64
from io import BytesIO
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model

from colorama import Fore, Style, init

from fastapi.middleware.cors import CORSMiddleware

import logging
tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(logging.ERROR)

# Inicializações
init(autoreset=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Configurações de diretórios e modelo
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'pokemon_model.h5')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

# Parâmetros da API
IMG_SIZE = (160, 160)
MIN_CONFIDENCE = 0.5

# Logs de inicialização
print(f"{Fore.GREEN}🔧 Iniciando servidor...{Style.RESET_ALL}")
print(f"{Fore.CYAN}📂 Caminho do modelo: {MODEL_PATH}{Style.RESET_ALL}")
print(f"{Fore.CYAN}📂 Diretório do dataset: {DATASET_DIR}{Style.RESET_ALL}")

# Carregamento do modelo e classes
def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"{Fore.RED}ERRO: Modelo não encontrado em {MODEL_PATH}")
        sys.exit("Execute o treinamento antes de iniciar a API.")
    return keras_load_model(MODEL_PATH)

def get_class_names():
    if not os.path.exists(DATASET_DIR):
        print(f"{Fore.RED}ERRO: Dataset não encontrado em {DATASET_DIR}{Style.RESET_ALL}")
        sys.exit("Verifique a existência do diretório do dataset.")
    return sorted(os.listdir(DATASET_DIR))

MODEL = load_model()
CLASS_NAMES = get_class_names()

# Modelo Pydantic
class ImageData(BaseModel):
    image: str

# Funções de processamento
def base64_to_image(b64_string: str) -> Image.Image:
    try:
        image_data = base64.b64decode(b64_string)
        image = Image.open(BytesIO(image_data))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Erro ao converter base64 em imagem: {str(e)}")

def preprocess_image(image: Image.Image) -> np.ndarray:
    try:
        image = image.resize(IMG_SIZE)
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise ValueError(f"Erro no pré-processamento da imagem: {str(e)}")

def postprocess_prediction(predictions: np.ndarray) -> str:
    confidence = np.max(predictions)
    if confidence < MIN_CONFIDENCE:
        raise ValueError(f"Confiança insuficiente: {confidence:.2f}")
    index = int(np.argmax(predictions))
    pokemon_name = CLASS_NAMES[index]
    return pokemon_name

# Aplicação FastAPI
app = FastAPI(title="API de Predição de Pokémon", description="Recebe imagem em base64, converte para PNG e retorna o nome do Pokémon.")

@app.post("/predict")
async def predict(image_data: ImageData):
    try:
        print(f"{Fore.YELLOW}📥 Recebendo imagem (base64 length: {len(image_data.image)})...{Style.RESET_ALL}")
        image = base64_to_image(image_data.image)
        print(f"{Fore.GREEN}✅ Imagem convertida (Tamanho: {image.size}){Style.RESET_ALL}")
    except ValueError as e:
        print(f"{Fore.RED}❌ Erro na conversão da imagem: {str(e)}{Style.RESET_ALL}")
        raise HTTPException(status_code=400, detail=str(e))
    
    try:
        print(f"{Fore.YELLOW}🖼️ Processando imagem (Redimensionando para {IMG_SIZE})...{Style.RESET_ALL}")
        processed_image = preprocess_image(image)
        print(f"{Fore.GREEN}✅ Imagem processada (Shape: {processed_image.shape}){Style.RESET_ALL}")
    except ValueError as e:
        print(f"{Fore.RED}❌ Erro no pré-processamento: {str(e)}{Style.RESET_ALL}")
        raise HTTPException(status_code=400, detail=str(e))
    
    try:
        print(f"{Fore.YELLOW}🤖 Realizando predição...{Style.RESET_ALL}")
        predictions = MODEL.predict(processed_image)
        print(f"{Fore.GREEN}✅ Predição concluída (Confiança: {np.max(predictions):.2f}){Style.RESET_ALL}")
        pokemon_name = postprocess_prediction(predictions)
        print(f"{Fore.BLUE}🎉 Resultado: {pokemon_name} (Confiança: {np.max(predictions):.2f}){Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}❌ Erro na predição: {str(e)}{Style.RESET_ALL}")
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")
    
    return {"status": "sucesso", "pokemon": pokemon_name}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas origens
    allow_methods=["POST"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    print(f"{Fore.MAGENTA}🚀 Servidor pronto em http://0.0.0.0:8000{Style.RESET_ALL}")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)