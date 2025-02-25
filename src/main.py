import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
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
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
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
model = Sequential([
    EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.3),  # Regularização adicional
    Dense(len(pokemon_list), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=3e-4),  # Taxa de aprendizado ajustada
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

print_status("Avaliação final...", "📝")
loss, accuracy = model.evaluate(val_data)
print(f"{Fore.GREEN}🎯 Acurácia final: {accuracy*100:.2f}%")

print_status("Salvando modelo...", "💾")
model.save('pokemon_model.keras')
print(f"{Fore.GREEN}✅ Treinamento concluído!")