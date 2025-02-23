# Importações necessárias
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import load_img
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import warnings
import shutil

# Configuração inicial do ambiente
warnings.filterwarnings("ignore", category=UserWarning)

# Configuração de GPU/CPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU detectada e configurada!")
else:
    print("Usando CPU")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduz logs do TensorFlow

# Função para carregar imagens com tratamento de erros
def safe_load_img(path, target_size=(224, 224)):
    try:
        img = load_img(path, target_size=target_size)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        print(f"Removendo arquivo inválido: {path}")
        os.remove(path)
        return None

# Função para verificar integridade do dataset
def verify_dataset(dataset_path='dataset'):
    shutil.rmtree(os.path.join(dataset_path, '.ipynb_checkpoints'), ignore_errors=True)
    
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            print(f"{class_name}: {len(os.listdir(class_path))} imagens")

verify_dataset()

# Parâmetros fundamentais do modelo
IMG_SIZE = 224
BATCH_SIZE = 32

# Configuração de aumento de dados e pré-processamento
data_gen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.9,1.1],
    validation_split=0.2,
    dtype='float32'
)

# Remoção de artefatos temporários
shutil.rmtree('dataset/.ipynb_checkpoints', ignore_errors=True)

# Carregamento dos dados de treino
train_data = data_gen.flow_from_directory(
    'dataset',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    color_mode='rgba' if os.listdir('dataset')[0].endswith('.png') else 'rgb',
    classes=None
)

# Carregamento dos dados de validação
val_data = data_gen.flow_from_directory(
    'dataset',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    color_mode='rgba' if os.listdir('dataset')[0].endswith('.png') else 'rgb',
    classes=None
)

# Análise de distribuição das classes
print("\nDistribuição de classes:")
print(np.unique(train_data.classes, return_counts=True))

# Cálculo de pesos para classes desbalanceadas
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))

# Construção da arquitetura do modelo
base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 4 if train_data.image_shape[2] == 4 else 3)
)

# Estratégia de transfer learning
base_model.trainable = False
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Definição das camadas do modelo
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dense(len(train_data.class_indices), activation='softmax')
])

# Configuração do processo de otimização
model.compile(
    optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Configuração de callbacks para treino
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', save_best_only=True),
    ReduceLROnPlateau(factor=0.2, patience=2)
]

# Fase 1: Treinamento inicial
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=callbacks,
    class_weight=class_weights
)

# Fase 2: Fine-tuning do modelo
if len(history.history['val_loss']) >= 5:
    base_model.trainable = True
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    history_fine = model.fit(
        train_data,
        validation_data=val_data,
        epochs=10,
        callbacks=callbacks
    )

# Avaliação final do modelo
loss, accuracy = model.evaluate(val_data)
print(f"\nAcurácia final: {accuracy*100:.2f}%")

# Salvamento dos resultados
model.save('pokemon_model.keras')
with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)