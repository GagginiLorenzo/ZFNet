import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from base import *
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint

base_dir = 'data'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
# Dimensions des images
img_width, img_height = 224, 224

# Augmentation des données et prétraitement
train_datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)

checkpoint_callback = ModelCheckpoint(
    filepath='zfnet_model_callback.keras',  # Sauvegarder le modèle avec le numéro d'époque dans le nom de fichier
    save_weights_only=False,  # Sauvegarder l'ensemble du modèle (architecture + poids + état de l'optimiseur)
    save_freq='epoch'  # Sauvegarder le modèle après chaque époque
)

# Nombre de classes
num_classes = len(train_generator.class_indices)
train_generator.class_indices
model = create_zfnet_model((img_width, img_height, 3), num_classes)
model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


batch_size = 32
epochs = 1

# J'ai' une boucle de 1 à 9 époques pour éviter des problèmes de stabilité de la machine
# Au lieu de faire directement 10 époques d'entraînement.
# Le callback ModelCheckpoint devait suffire pour sauvegarder le modèle à chaque époque.
# Mais j'ai eu des erreurs de segmentation lors de l'entraînement en faisant 10 époques d'un coup.
# j'ai donc décidé de faire une boucle de 1 à 9 époques pour compléter les 10 époques sans erreurs.

for i in range(1, 10):
    print(i)
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=epochs,
        callbacks=[checkpoint_callback]
    )
    model.save(f'zfnet_model_epoch_{i}.keras')


model.save('10.h5')
model.save('10.keras')

test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f'Test accuracy: {test_accuracy:.4f}')

# Charger le modèle sauvegardé pour vérifier qu'il a bien été sauvegardé
model_load = tf.keras.models.load_model('10.keras')

# Vérifier le modèle en imprimant son résumé
model_load.summary()

test_loss, test_accuracy = model_load.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f'Test accuracy: {test_accuracy:.4f}')
