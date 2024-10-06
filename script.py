from base import *
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input

input_shape = (224, 224, 3)  # Format
num_classes = 2  # Nombre de classes

model = create_zfnet_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Quel optimiseur utiliser ?
model.summary()


# Charger et prétraiter une image
img_path = 'bunny.jpeg'
img_array = load_and_preprocess_image(img_path, target_size=(224, 224))
plt.imshow(img_array[0])
names = [layer.name for layer in model.layers]
layer_outputs = [model.get_layer(name).output for name in names]
names # Noms des couches du modèle

activations = get_activations(model, img_array, names[1]) # Obtenir les activations de la n-ième couche

# Visualiser les filtres
num_filters = 4  # Nombre de filtres à visualiser
visualize_filters(activations[1], num_filters) # Visualiser les filtres convoluer (activations[1] -> tableau convolué)

activations[0].get_config() # (actiavtions[0] -> couche de convolution)
