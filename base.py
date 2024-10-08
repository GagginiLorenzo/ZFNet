import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import ReLU, Conv2D, MaxPooling2D, Dense, Flatten, Dropout,Input,Conv2DTranspose,UpSampling2D
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


       # Stack connectée 2
       model.add(Dense(4096, activation='relu'))


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
def visualize_filters(activations, num_filters,name):
    fig, axes = plt.subplots(1, num_filters, figsize=(20, 20))
    for i in range(num_filters):
        ax = axes[i]
        ax.imshow(activations[0, :, :, i], cmap='viridis')
        ax.axis('off')
    plt.savefig(name)
    plt.show()

# Fonction pour évaluer le modèle sur une image en plus jolie
def evaluate_model_on_image(img, description, model):
    img_array = load_and_preprocess_image(img, target_size=(224, 224))
    #plt.imshow(img_array[0])
    prediction = model.predict(img_array)
    print(f"Prediction for {description}: {prediction}")
    return prediction


############################################################################################################ à faire
# Attacher les couches de déconvolutions (REUSSI!) reste a généraliser pour tout les stacks
# Entrainement du modèle (ImageNet)
# Test visuel et scoring (à définir)

def create_deconv_test(input_shape):
    """
    DECONV QUE LE PREMIER STACK POUR LE MOMENT !!
    """
    model = Sequential()
    model.add(Input(shape=input_shape))

    # Upsampling to reverse the MaxPooling2D layer
    model.add(UpSampling2D(size=(2, 2))) # surement pas la bonne methode (verifier le type de pooling)
    # ReLU activation
    model.add(ReLU())
    # Deconvolution to reverse the Conv2D layer
    model.add(Conv2DTranspose(96, (7, 7), strides=(2, 2), activation='relu')) # dernier activation relu ? à verifier...

    return model

def create_deconv_model(input_shape, stack):

    model = Sequential()
    model.add(Input(shape=input_shape))
    match stack:
        case 5:
            # Cinquième stack
            model.add(UpSampling2D(size=(2, 2)))
            model.add(ReLU())
            model.add(Conv2DTranspose(384, (3, 3)))

            # Quatrième stack
            model.add(Conv2DTranspose(384, (3, 3)))

            # Troisième stack
            model.add(Conv2DTranspose(256, (3, 3)))

            # Deuxième stack
            model.add(UpSampling2D(size=(2, 2)))
            model.add(ReLU())
            model.add(Conv2DTranspose(96, (5, 5), strides=(2, 2)))

            # Première stack
            model.add(UpSampling2D(size=(2, 2)))
            model.add(ReLU())
            model.add(Conv2DTranspose(3, (7, 7), strides=(2, 2)))
        case 4:
            # Quatrième stack
            model.add(Conv2DTranspose(384, (3, 3)))

            # Troisième stack
            model.add(Conv2DTranspose(256, (3, 3)))

            # Deuxième stack
            model.add(UpSampling2D(size=(2, 2)))
            model.add(ReLU())
            model.add(Conv2DTranspose(96, (5, 5), strides=(2, 2)))

            # Première stack
            model.add(UpSampling2D(size=(2, 2)))
            model.add(ReLU())
            model.add(Conv2DTranspose(3, (7, 7), strides=(2, 2)))
        case 3:
            # Troisième stack
            model.add(Conv2DTranspose(256, (3, 3)))

            # Deuxième stack
            model.add(UpSampling2D(size=(2, 2)))
            model.add(ReLU())
            model.add(Conv2DTranspose(96, (5, 5), strides=(2, 2)))

            # Première stack
            model.add(UpSampling2D(size=(2, 2)))
            model.add(ReLU())
            model.add(Conv2DTranspose(3, (7, 7), strides=(2, 2)))
        case 2:
            # Deuxième stack
            model.add(UpSampling2D(size=(2, 2)))
            model.add(ReLU())
            model.add(Conv2DTranspose(96, (5, 5), strides=(2, 2)))

            # Première stack
            model.add(UpSampling2D(size=(2, 2)))
            model.add(ReLU())
            model.add(Conv2DTranspose(3, (7, 7), strides=(2, 2)))
        case 1:
            # Première stack
            model.add(UpSampling2D(size=(2, 2)))
            model.add(ReLU())
            model.add(Conv2DTranspose(3, (7, 7), strides=(2, 2)))
    model.summary()
    return model

def alt_create_deconv_model(input_shape, stack):

    model = Sequential()
    model.add(Input(shape=input_shape))
    match stack:
        case 5:
            # Cinquième stack
            model.add(UpSampling2D(size=(2, 2)))
            model.add(ReLU())
            model.add(Conv2DTranspose(384, (3, 3)))

            # Quatrième stack
            model.add(UpSampling2D(size=(2, 2)))
            model.add(ReLU())
            model.add(Conv2DTranspose(384, (3, 3)))

            # Troisième stack
            model.add(UpSampling2D(size=(2, 2)))
            model.add(ReLU())
            model.add(Conv2DTranspose(256, (3, 3)))

            # Deuxième stack
            model.add(UpSampling2D(size=(2, 2)))
            model.add(ReLU())
            model.add(Conv2DTranspose(96, (5, 5), strides=(2, 2)))

            # Première stack
            model.add(UpSampling2D(size=(2, 2)))
            model.add(ReLU())
            model.add(Conv2DTranspose(3, (7, 7), strides=(2, 2), activation='relu'))
        case 4:
            # Quatrième stack
            model.add(UpSampling2D(size=(2, 2)))
            model.add(ReLU())
            model.add(Conv2DTranspose(384, (3, 3)))

            # Troisième stack
            model.add(UpSampling2D(size=(2, 2)))
            model.add(ReLU())
            model.add(Conv2DTranspose(256, (3, 3)))

            # Deuxième stack
            model.add(UpSampling2D(size=(2, 2)))
            model.add(ReLU())
            model.add(Conv2DTranspose(96, (5, 5), strides=(2, 2)))

            # Première stack
            model.add(UpSampling2D(size=(2, 2)))
            model.add(ReLU())
            model.add(Conv2DTranspose(3, (7, 7), strides=(2, 2), activation='relu'))
        case 3:
            # Troisième stack
            model.add(UpSampling2D(size=(2, 2)))
            model.add(ReLU())
            model.add(Conv2DTranspose(256, (3, 3)))

            # Deuxième stack
            model.add(UpSampling2D(size=(2, 2)))
            model.add(ReLU())
            model.add(Conv2DTranspose(96, (5, 5), strides=(2, 2)))

            # Première stack
            model.add(UpSampling2D(size=(2, 2)))
            model.add(ReLU())
            model.add(Conv2DTranspose(3, (7, 7), strides=(2, 2), activation='relu'))
        case 2:
            # Deuxième stack
            model.add(UpSampling2D(size=(2, 2)))
            model.add(ReLU())
            model.add(Conv2DTranspose(96, (5, 5), strides=(2, 2)))

            # Première stack
            model.add(UpSampling2D(size=(2, 2)))
            model.add(ReLU())
            model.add(Conv2DTranspose(3, (7, 7), strides=(2, 2), activation='relu'))
        case 1:
            # Première stack
            model.add(UpSampling2D(size=(2, 2)))
            model.add(ReLU())
            model.add(Conv2DTranspose(3, (7, 7), strides=(2, 2), activation='relu'))
    model.summary()
    return model
