import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from colorama import Fore, Style, init
from tqdm import tqdm
from yaspin import yaspin
import os
import warnings

# Inicializa colorama
init(autoreset=True)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def run_cross_validation():
    # Configurações otimizadas
    IMG_SIZE = 160
    BATCH_SIZE = 32
    EPOCHS = 15
    N_SPLITS = 5
    ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

    print(f"\n{Fore.CYAN}=== INICIANDO VALIDAÇÃO CRUZADA ({N_SPLITS} FOLDS) ==={Style.RESET_ALL}")

    # Carregamento de dados com verificação reforçada
    base_dir = 'dataset'
    class_names = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    if not class_names:
        print(f"{Fore.RED}Erro: Nenhuma classe encontrada no diretório {base_dir}!{Style.RESET_ALL}")
        return

    file_paths, labels = [], []
    
    # Coleta de arquivos com tratamento de erros
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"{Fore.RED}Diretório inválido: {class_dir}{Style.RESET_ALL}")
            continue
            
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if any(img_name.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
                file_paths.append(img_path)
                labels.append(class_idx)
            else:
                print(f"{Fore.YELLOW}Aviso: Formato não suportado - {img_name}{Style.RESET_ALL}")

    if not file_paths:
        print(f"{Fore.RED}Erro: Nenhuma imagem válida encontrada!{Style.RESET_ALL}")
        return

    # Verificação de integridade das imagens com spinner e tqdm
    print(f"\n{Fore.YELLOW}Verificando integridade das imagens...{Style.RESET_ALL}")
    valid_indices = []
    with yaspin(text=f"{Fore.BLUE}Checando arquivos...", color="cyan") as spinner:
        for idx in tqdm(range(len(file_paths)), desc="Verificando imagens", ncols=80):
            try:
                img = tf.io.read_file(file_paths[idx])
                tf.image.decode_image(img, channels=3)
                valid_indices.append(idx)
            except Exception as e:
                print(f"\n{Fore.RED}Removendo arquivo corrompido: {file_paths[idx]} - Erro: {str(e)}{Style.RESET_ALL}")
        spinner.ok("✔")
    
    file_paths = np.array(file_paths)[valid_indices]
    labels = np.array(labels)[valid_indices]

    # Configuração do K-Fold
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_accuracies = []

    # Pipeline de pré-processamento aprimorado
    def preprocess_image(image_path, label):
        try:
            image = tf.io.read_file(image_path)
            
            # Decodificação específica para PNG usando string raw
            if tf.strings.regex_full_match(image_path, r".*\.png"):
                image = tf.io.decode_png(image, channels=3)
            else:
                image = tf.image.decode_image(image, channels=3, expand_animations=False)
            
            # Aumento de dados otimizado
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
            image = tf.image.resize(image, [IMG_SIZE + 20, IMG_SIZE + 20])
            image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])
            
            return tf.keras.applications.efficientnet.preprocess_input(image), label
        except Exception as e:
            print(f"{Fore.RED}Erro no pré-processamento: {image_path} - {str(e)}{Style.RESET_ALL}")
            # Em caso de erro, retornamos tensor de zeros e label sentinela (-1)
            return tf.zeros([IMG_SIZE, IMG_SIZE, 3]), tf.constant(-1, dtype=tf.int32)

    # Filtro otimizado: filtra exemplos cuja label seja diferente de -1
    def filter_invalid(image, label):
        return tf.not_equal(label, -1)

    # Configuração do dataset
    def configure_dataset(ds):
        return ds.cache().prefetch(tf.data.AUTOTUNE)

    # Loop de validação cruzada
    for fold, (train_idx, val_idx) in enumerate(kf.split(file_paths)):
        tf.keras.backend.clear_session()  # Limpeza de memória
        
        print(f"\n{Fore.CYAN}=== Fold {fold+1}/{N_SPLITS} ==={Style.RESET_ALL}")

        # Pipeline de dados robusto
        train_dataset = (
            tf.data.Dataset.from_tensor_slices((file_paths[train_idx], labels[train_idx]))
            .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
            .filter(filter_invalid)
            .apply(tf.data.experimental.ignore_errors())
            .batch(BATCH_SIZE)
            .apply(configure_dataset)
        )

        val_dataset = (
            tf.data.Dataset.from_tensor_slices((file_paths[val_idx], labels[val_idx]))
            .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
            .filter(filter_invalid)
            .apply(tf.data.experimental.ignore_errors())
            .batch(BATCH_SIZE)
            .apply(configure_dataset)
        )

        # Modelo com fine-tuning
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        base_model.trainable = True

        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(512, activation='relu', 
                                  kernel_regularizer=tf.keras.regularizers.l2(0.02)),
            tf.keras.layers.Dropout(0.6),
            tf.keras.layers.Dense(len(class_names), activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Spinner para o treinamento de cada fold
        with yaspin(text=f"{Fore.MAGENTA}Treinando fold {fold+1}... 🛠️", color="magenta") as train_spinner:
            try:
                history = model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=EPOCHS,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            patience=5, 
                            restore_best_weights=True,
                            monitor='val_loss'
                        ),
                        tf.keras.callbacks.ReduceLROnPlateau(
                            factor=0.5, 
                            patience=2,
                            verbose=1
                        ),
                        tf.keras.callbacks.ModelCheckpoint(
                            f'fold_{fold+1}_model.keras',
                            save_best_only=True,
                            monitor='val_accuracy'
                        )
                    ],
                    verbose=0
                )
                # Finaliza o spinner com sucesso
                train_spinner.ok("✔")
            except Exception as e:
                train_spinner.fail("✖")
                print(f"{Fore.RED}Erro no treinamento: {str(e)}{Style.RESET_ALL}")
                continue

        # Avaliação precisa
        _, val_acc = model.evaluate(val_dataset, verbose=0)
        fold_accuracies.append(val_acc)
        print(f"{Fore.GREEN}Acurácia Fold {fold+1}: {val_acc*100:.2f}% {Style.RESET_ALL}")

    # Resultados finais com mensagem decorada
    if fold_accuracies:
        print(f"\n{Fore.CYAN}=== RESULTADOS ==={Style.RESET_ALL}")
        print(f"🎉 Acurácia média: {np.mean(fold_accuracies)*100:.2f}%")
        print(f"📊 Desvio padrão: {np.std(fold_accuracies)*100:.2f}%{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Todos os folds falharam!{Style.RESET_ALL}")

if __name__ == "__main__":
    run_cross_validation()