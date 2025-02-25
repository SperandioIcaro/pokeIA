import os
from PIL import Image
from tqdm import tqdm
from colorama import Fore, Style, init
from yaspin import yaspin

# Configuração inicial
init(autoreset=True)

def print_colored(message, color=Fore.WHITE, emoji=''):
    print(f"{color}{emoji} {message}{Style.RESET_ALL}")

def clean_dataset(dataset_path='dataset'):
    corrupted = 0
    total_files = 0
    
    with yaspin(text=f"{Fore.BLUE}Verificando estrutura do dataset...", color="magenta") as spinner:
        for root, _, files in os.walk(dataset_path):
            total_files += len(files)
        spinner.ok("✔")
    
    progress_bar = tqdm(
        total=total_files,
        desc=f"{Fore.CYAN}Limpando arquivos",
        bar_format="{l_bar}{bar:20}{r_bar}",
        colour='GREEN'
    )
    
    for root, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                progress_bar.set_postfix_str(f"Processando: {file[:15]}...")
                
                if file.startswith('.'):
                    os.remove(file_path)
                    print_colored(f"Arquivo oculto removido: {file}", Fore.YELLOW, '🗑️')
                    corrupted += 1
                    continue
                    
                with Image.open(file_path) as img:
                    img.verify()
                
                if file_path.endswith('.png'):
                    with Image.open(file_path) as img:
                        if img.mode not in ['RGB', 'RGBA']:
                            img.convert('RGBA').save(file_path)
                            print_colored(f"Arquivo convertido: {file}", Fore.BLUE, '🔄')
                
            except Exception as e:
                print_colored(f"Arquivo corrompido: {file} | Erro: {str(e)}", Fore.RED, '❌')
                os.remove(file_path)
                corrupted += 1
            
            progress_bar.update(1)
    
    progress_bar.close()
    print_colored(f"\nRemovidos {corrupted} arquivos problemáticos", Fore.GREEN, '✅')

if __name__ == "__main__":
    clean_dataset()