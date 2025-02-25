import requests
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
from colorama import Fore, Style, init
from yaspin import yaspin

# Inicializa colorama
init(autoreset=True)

def print_colored(message, color=Fore.WHITE, emoji=''):
    """Exibe mensagens formatadas com cores e emojis"""
    print(f"{color}{emoji} {message}{Style.RESET_ALL}")

def get_existing_pokemon(base_dir='dataset'):
    return [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

def should_skip_pokemon(pokemon, min_images, base_dir='dataset'):
    dir_path = os.path.join(base_dir, pokemon)
    if not os.path.exists(dir_path):
        return False
    image_count = len([f for f in os.listdir(dir_path) if f.endswith('.png') and '_aug' not in f])
    return image_count >= min_images

def get_all_sprites(data):
    urls = []
    def extract_urls(obj):
        if isinstance(obj, dict):
            for _, value in obj.items():
                extract_urls(value)
        elif isinstance(obj, list):
            for item in obj:
                extract_urls(item)
        elif isinstance(obj, str) and obj.endswith('.png'):
            urls.append(obj)
    extract_urls(data['sprites'])
    return list(set(urls))

def augment_image(img_path, output_dir, num_augmented=3):
    try:
        img = Image.open(img_path)
        base_name = os.path.basename(img_path)
        name, ext = os.path.splitext(base_name)
        
        with yaspin(text=f"  Gerando aumentações para {base_name}", color="yellow") as spinner:
            for i in range(num_augmented):
                angle = np.random.randint(-20, 20)
                zoom = np.random.uniform(0.8, 1.2)
                
                new_img = img.rotate(angle, expand=True)
                new_size = (int(new_img.width * zoom), int(new_img.height * zoom))
                new_img = new_img.resize(new_size, Image.LANCZOS)
                
                new_name = f"{name}_aug{i}{ext}"
                new_img.save(os.path.join(output_dir, new_name))
                spinner.text = f"  Gerada {i+1}/{num_augmented} aumentações"
            
            spinner.ok("✔")

    except Exception as e:
        print_colored(f"Erro ao aumentar {img_path}: {str(e)}", Fore.RED, '⚠️')

def download_pokemon_images(pokemon_list, min_images=120):
    base_dir = os.path.abspath('dataset')
    os.makedirs(base_dir, exist_ok=True)
    
    stats = {
        'total': 0,
        'downloaded': 0,
        'augmented': 0
    }

    with tqdm(pokemon_list, desc=f"{Fore.CYAN}Processando Pokémon{Style.RESET_ALL}", 
             bar_format="{l_bar}{bar:20}{r_bar}") as pbar:
        for pokemon in pbar:
            try:
                dir_path = os.path.join(base_dir, pokemon)
                pbar.set_postfix_str(f"{Fore.YELLOW}{pokemon}{Style.RESET_ALL}")
                
                if should_skip_pokemon(pokemon, min_images, base_dir):
                    print_colored(f"{pokemon} já completo - pulando", Fore.GREEN, '✅')
                    continue
                    
                os.makedirs(dir_path, exist_ok=True)
                
                # Download das sprites
                with yaspin(text=f"  Buscando dados na API", color="cyan") as spinner:
                    response = requests.get(f"https://pokeapi.co/api/v2/pokemon/{pokemon.lower()}", timeout=15)
                    data = response.json()
                    urls = get_all_sprites(data)
                    spinner.ok("✔")

                existing_files = os.listdir(dir_path)
                new_downloads = 0
                
                with tqdm(urls, desc=f"  {Fore.BLUE}Baixando sprites{Style.RESET_ALL}", 
                         leave=False, bar_format="{l_bar}{bar:10}{r_bar}") as dl_bar:
                    for idx, url in enumerate(dl_bar):
                        filename = f"{idx:03d}.png"
                        if filename not in existing_files:
                            img_data = requests.get(url, timeout=10).content
                            with open(os.path.join(dir_path, filename), 'wb') as f:
                                f.write(img_data)
                            new_downloads += 1
                        dl_bar.set_postfix_str(f"Novos: {new_downloads}")
                
                stats['downloaded'] += new_downloads
                
                # Augmentação
                original_images = [f for f in os.listdir(dir_path) if f.endswith('.png') and '_aug' not in f]
                augmented = 0
                
                with tqdm(total=min_images, desc=f"  {Fore.MAGENTA}Gerando aumentações{Style.RESET_ALL}", 
                         initial=len(os.listdir(dir_path)), leave=False) as aug_bar:
                    while len(os.listdir(dir_path)) < min_images and original_images:
                        needed = min_images - len(os.listdir(dir_path))
                        per_image = max(needed // len(original_images) + 1, 1)
                        
                        for img_file in original_images:
                            augment_image(
                                os.path.join(dir_path, img_file),
                                dir_path,
                                num_augmented=per_image
                            )
                            new_files = len(os.listdir(dir_path))
                            aug_bar.update(new_files - aug_bar.n)
                            augmented += per_image
                            if new_files >= min_images:
                                break
                
                stats['augmented'] += augmented
                stats['total'] += 1
                print_colored(f"Concluído! Total: {len(os.listdir(dir_path))} imagens", Fore.GREEN, '🎉')

            except requests.exceptions.RequestException as e:
                print_colored(f"Erro de rede: {str(e)}", Fore.RED, '⚠️')
            except Exception as e:
                print_colored(f"Erro crítico: {str(e)}", Fore.RED, '❌')

    # Resumo final
    print(f"\n{Fore.CYAN}=== Resumo Final ==={Style.RESET_ALL}")
    print(f"{Fore.GREEN}Pokémon processados: {stats['total']}")
    print(f"{Fore.BLUE}Imagens baixadas: {stats['downloaded']}")
    print(f"{Fore.MAGENTA}Variações geradas: {stats['augmented']}{Style.RESET_ALL}")

if __name__ == "__main__":
    base_dir = os.path.abspath('dataset')
    pokemon_list = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    if not pokemon_list:
        pokemon_list = [
            'aegislash-blade', 'aegislash-shield',
            'abomasnow', 'aggron', 'accelgor',
            'aerodactyl', 'aipom', 'alakazam', 'absol', 'abra'
        ]
    
    download_pokemon_images(pokemon_list, min_images=120)