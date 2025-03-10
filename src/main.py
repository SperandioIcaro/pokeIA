import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from colorama import Fore, Style, init
from tqdm.keras import TqdmCallback

init(autoreset=True)
warnings.filterwarnings("ignore")

def print_status(message, emoji="🔄"):
    print(f"\n{Fore.CYAN}{emoji} {Fore.WHITE}{message}")

class ColorProgress(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        acc = logs['accuracy']
        val_acc = logs['val_accuracy']
        print(f"{Fore.GREEN}Epoch {epoch+1:02d} | Acc: {acc:.2f} {Fore.YELLOW}| Val Acc: {val_acc:.2f}")

print_status("Configurando ambiente...", "⚙️")
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"{Fore.GREEN}✅ GPU configurada")
else:
    print(f"{Fore.YELLOW}⚠️  Usando CPU")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print_status("Analisando dataset...", "📁")
def verify_dataset(path='dataset'):
    classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for cls in classes:
        count = len(os.listdir(os.path.join(path, cls)))
        print(f"{Fore.MAGENTA}➤ {cls}: {Fore.WHITE}{count} imagens")
    return classes

pokemon_list = verify_dataset()

IMG_SIZE = 160  # Aumento leve na resolução
BATCH_SIZE = 32  # Aumento controlado no batch size

print_status("Preparando dados...", "📊")
train_gen = ImageDataGenerator(
    rotation_range=25,  # Aumentado de 15 para 25
    zoom_range=0.2,     # Aumentado de 0.1 para 0.2
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],  # Novo
    # contrast_range=[0.8, 1.2],    # Novo
    horizontal_flip=True,         # Novo
    validation_split=0.2,
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

train_data = train_gen.flow_from_directory(
    'dataset',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    color_mode='rgb'
)

val_data = train_gen.flow_from_directory(
    'dataset',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    color_mode='rgb'
)

print_status("Construindo modelo...", "🧠")
# Adicione na seção de data augmentation:
train_gen = ImageDataGenerator(
    rotation_range=25,  # Aumentado de 15 para 25
    zoom_range=0.2,     # Aumentado de 0.1 para 0.2
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],  # Novo
    # contrast_range=[0.8, 1.2],    # Novo
    horizontal_flip=True,         # Novo
    validation_split=0.2,
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

# Modifique a construção do modelo:
base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet', 
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = True  # Descongelar camadas para fine-tuning

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)),  # Aumentado
    Dropout(0.6),  # Aumentado
    Dense(len(pokemon_list), activation='softmax')
])

# Ajuste o otimizador:
model.compile(
    optimizer=Adam(learning_rate=1e-5),  # Reduzido para fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', save_best_only=True),
    ReduceLROnPlateau(factor=0.5, patience=2),
    ColorProgress(),
    TqdmCallback(verbose=0)  # Barra de progresso leve
]

print_status("Iniciando treinamento...", "🚀")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,  # Epochs aumentadas
    callbacks=callbacks,
    verbose=0
)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Acurácia Treino')
plt.plot(history.history['val_accuracy'], label='Acurácia Validação')
plt.title('Evolução da Acurácia')
plt.ylabel('Acurácia')
plt.xlabel('Época')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Perda Treino')
plt.plot(history.history['val_loss'], label='Perda Validação')
plt.title('Evolução da Perda')
plt.ylabel('Perda')
plt.xlabel('Época')
plt.legend()

plt.tight_layout()
plt.show()

print_status("Avaliação final...", "📝")
loss, accuracy = model.evaluate(val_data)
print(f"{Fore.GREEN}🎯 Acurácia final: {accuracy*100:.2f}%")

print_status("Salvando modelo...", "💾")
model.save('pokemon_model.keras')
print(f"{Fore.GREEN}✅ Treinamento concluído!")