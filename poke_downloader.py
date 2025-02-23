import requests
import os
from tqdm import tqdm
from PIL import Image
import numpy as np

# Função para aumento de dados
def augment_image(img_path, output_dir, num_augmented=5):
    img = Image.open(img_path)
    for i in range(num_augmented):
        angle = np.random.randint(-15, 15)
        zoom = np.random.uniform(0.9, 1.1)
        new_img = img.rotate(angle).resize((int(img.width*zoom), int(img.height*zoom)))
        new_img.save(os.path.join(output_dir, f'augmented_{i}_{os.path.basename(img_path)}'))

# Função principal de download de imagens
def download_pokemon_images(pokemon_list, min_images=50):
    for pokemon in tqdm(pokemon_list, desc="Processando Pokémon"):
        try:
            class_dir = f'dataset/{pokemon}'
            os.makedirs(class_dir, exist_ok=True)
            
            # Download de sprites via PokeAPI
            response = requests.get(f"https://pokeapi.co/api/v2/pokemon/{pokemon}")
            data = response.json()
            
            count = 0
            for key, url in data['sprites'].items():
                if isinstance(url, str) and url.endswith('.png'):
                    img_data = requests.get(url).content
                    img_path = os.path.join(class_dir, f'sprite_{count}.png')
                    with open(img_path, 'wb') as f:
                        f.write(img_data)
                    count += 1
            
            # Geração de variações sintéticas
            existing_images = [os.path.join(class_dir, f) for f in os.listdir(class_dir)]
            if len(existing_images) < min_images:
                for img_file in existing_images:
                    augment_image(img_file, class_dir, num_augmented=3)
                    
        except Exception as e:
            print(f"Erro em {pokemon}: {e}")

# Lista de Pokémon e execução
pokemon_list = [
    'aegislash-blade', 'aegislash-shield',
    'abomasnow', 'aggron', 'accelgor', 
    'aerodactyl', 'aipom', 'alakazam', 'absol', 'abra'
]

download_pokemon_images(pokemon_list, min_images=50)