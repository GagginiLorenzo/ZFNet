import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,Input,Conv2DTranspose
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image


def create_zfnet_model(input_shape, num_classes):

       model = Sequential()
       model.add(Input(shape=input_shape))

       # Première stack
       model.add(Conv2D(96, (7, 7), strides=(2, 2), activation='relu'))
       model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

       # Deuxième stack
       model.add(Conv2D(256, (5, 5), strides=(2, 2), activation='relu'))
       model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

       # Troisième stack
       model.add(Conv2D(384, (3, 3), activation='relu'))

       # Quatrième stack
       model.add(Conv2D(384, (3, 3), activation='relu'))

       # Cinquième stack
       model.add(Conv2D(256, (3, 3), activation='relu'))
       model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

       # flattening
       model.add(Flatten())

       # Stack connectée 1
       model.add(Dense(4096, activation='relu'))
       model.add(Dropout(0.5))

       # Stack connectée 2
       model.add(Dense(4096, activation='relu'))
       model.add(Dropout(0.5))

       # Couche de sortie -> classification
       model.add(Dense(num_classes, activation='softmax'))

       return model


# Fonction pour charger et prétraiter une image
def load_and_preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalisation
    return img_array

# Fonction pour obtenir les filtres sortant d'une couches du modèle ZFNet
def get_activations(model, img_array, layer_name):
    intermediate_model = tf.keras.models.Model(
        model.inputs,model.get_layer(layer_name).output)
    activations = intermediate_model.predict(img_array)
    intermediate_model.summary()
    return [intermediate_model, activations] # Retourne le modèle et ces filtres

# Fonction pour visualiser les filtres sortant d'une couche du modèle ZFNet
def visualize_filters(activations, num_filters):
    fig, axes = plt.subplots(1, num_filters, figsize=(20, 20))
    for i in range(num_filters):
        ax = axes[i]
        ax.imshow(activations[0, :, :, i], cmap='viridis')
        ax.axis('off')
    plt.savefig('output.jpg')
    plt.show()


############################################################################################################ à faire
# Attacher les couches de déconvolutions (pas encore trouver de bonne structure algorithmique)
# Entrainement du modèle (ImageNet)
# Test visuel et scoring (à définir)
