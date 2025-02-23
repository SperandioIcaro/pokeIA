import os
from PIL import Image
from tqdm import tqdm

# Função principal de limpeza do dataset
def clean_dataset(dataset_path='dataset'):
    corrupted = 0
    for root, _, files in tqdm(os.walk(dataset_path), desc="Verificando imagens"):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.startswith('.'):
                    os.remove(file_path)
                    continue
                    
                # Verificação básica de integridade
                with Image.open(file_path) as img:
                    img.verify()
                
                # Conversão para formatos adequados
                if file_path.endswith('.png'):
                    with Image.open(file_path) as img:
                        if img.mode not in ['RGB', 'RGBA']:
                            img.convert('RGBA').save(file_path)
            except Exception as e:
                print(f"Arquivo corrompido/Inválido: {file_path} | Erro: {str(e)}")
                os.remove(file_path)
                corrupted += 1
                
    print(f"\nRemovidos {corrupted} arquivos problemáticos.")

# Ponto de entrada do script
if __name__ == "__main__":
    clean_dataset()