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
# Image dimensions
img_width, img_height = 224, 224

# Data augmentation and preprocessing
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
    filepath='zfnet_model_epoch_{epoch:02d}.keras',  # Save the model with the epoch number in the filename
    save_weights_only=False,  # Save the entire model (architecture + weights + optimizer state)
    save_freq='epoch'  # Save the model after every epoch
)

# Number of classes
num_classes = len(train_generator.class_indices)
train_generator.class_indices
model = create_zfnet_model((img_width, img_height, 3), num_classes)
model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

batch_size = 32
epochs = 1

for i in range (1, 10):
    print(i)
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=epochs,
        callbacks=[checkpoint_callback]
    )
    model.save('zfnet_model_epoch_{i}.keras')

model.save('20.h5')
model.save('20.keras')

test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f'Test accuracy: {test_accuracy:.4f}')


model_load = tf.keras.models.load_model('10.keras')

# Verify the model by printing its summary
model_load.summary()

test_loss, test_accuracy = model_load.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f'Test accuracy: {test_accuracy:.4f}')
