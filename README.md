# 🚀 PokeIA - Identificação de Pokémon com IA

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Project Status](https://img.shields.io/badge/status-in--development-yellow)

Este projeto tem como intuito inicial criar uma I.A. com objetivo de especialização no reconhecimento e identificação de Pokémon utilizando redes neurais profundas com TensorFlow/Keras.

## 📋 Pré-requisitos

- Python 3.8+
- pip
- Virtualenv (recomendado)
- Git

## 🛠 Configuração Rápida

```bash
# Clone o repositório
git clone git@github.com:SperandioIcaro/pokeIA.git

# Crie e ative o ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Instale as dependências
pip install -r requirements.txt
```

## 🏃 Execução do Projeto

```bash
# Treinar o modelo (GPU recomendada)
python main.py
```

```bash
# Gerar imagens aumentadas
python poke_downloader.py
```

```bash
# Limpeza de itens indesejados e padronização de imagens
python clean_dataset.py
```

## 🧠 Estrutura do Projeto

```bash
pokeIA/
├── dataset/               # Imagens de Pokémon
    └── pokemon/
        ├── pokemon1/
        │   ├── 001.png
        │   └── 001_aug1.png
        └── pokemon2/
            ├── 025.png
            └── 025_aug1.png
├── src/
│   ├── main.py            # Script principal de treinamento
│   └── poke_downloader.py # Gerenciamento de dataset
├── requirements.txt       # Dependências do projeto
└── README.md
```

## ✨ Funcionalidades Principais

Aumento de Dados (poke_downloader.py)

```bash
def augment_image(img_path, output_dir, num_augmented=5):
    # Implementa transformações aleatórias
    img = Image.open(img_path)
    # Rotação entre -25 e +25 graus
    # Zoom de até 20%
    # Salvamento das novas imagens
```

Arquitetura do Modelo (main.py)

    Base EfficientNetB0 pré-treinada

    Fine-tuning em duas fases

    Callbacks para Early Stopping e ajuste de LR

Gerenciamento de Dataset

    Verificação automática de imagens corrompidas

    Balanceamento de classes

    Pré-processamento otimizado

## 📦 Dependências Principais

```txt
tensorflow==2.12.0
pillow==9.5.0
numpy==1.24.3
scikit-learn==1.2.2
```

## 💡 Dicas Rápidas

1. Preparação do dataset:

```bash
# Acesse o link abaixo e baixe as imagens ja padronizadas
https://drive.google.com/drive/folders/1M1zAbCHRkCD40-6X3K7v3UMeW7tNLJoh?usp=sharing

# Cole a pasta dataset na raiz do projeto ou crie a pasta e adicione suas proprias imagens seguindo op padrão informado anteriormente
```

2. Para melhor performance:

```bash
   # Execute em uma GPU com CUDA habilitado
python main.py --batch_size 32 --epochs 50
```

## 📄 Licença

Este projeto está licenciado sob a MIT License.