import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import ReLU, Conv2D, MaxPooling2D, Dense, Flatten, Dropout,Input,Conv2DTranspose,UpSampling2D
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image


def create_zfnet_model(input_shape, num_classes):
    """
        Crée un modèle ZFNet.

        Args:
            input_shape (tuple): La forme de l'entrée du modèle (hauteur, largeur, canaux).
            num_classes (int): Le nombre de classes pour la classification.

        Returns:
            tensorflow.keras.Sequential: Le modèle ZFNet.
    """
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

def create_deconv_model(input_shape, stack):

    """
    Crée un modèle de déconvolution pour inverser les stacks du modèle ZFNet.

    Args:
        input_shape (tuple): La forme de l'entrée du modèle (hauteur, largeur, canaux).
        stack (int): Le numéro de la stack à inverser (1 à 5).

    Returns:
        tensorflow.keras.Sequential: Le modèle de déconvolution.
    """
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

# Fonction pour charger et prétraiter une image
def load_and_preprocess_image(img_path, target_size):
    """
    Charge et prétraite une image.

    Args:
        img_path (str): Le chemin de l'image à charger.
        target_size (tuple): La taille cible pour redimensionner l'image (hauteur, largeur).

    Returns:
        numpy.ndarray: L'image prétraitée sous forme de tableau numpy.
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalisation
    return img_array

# Fonction pour obtenir les filtres sortant d'une couches du modèle ZFNet
def get_activations(model, img_array, layer_name):
    """
    Obtient les activations d'une couche spécifique du modèle ZFNet.

    Args:
        model (tensorflow.keras.Model): Le modèle ZFNet.
        img_array (numpy.ndarray): L'image d'entrée sous forme de tableau numpy.
        layer_name (str): Le nom de la couche dont on veut obtenir les activations.

    Returns:
        tuple: Un tuple contenant le modèle intermédiaire et les activations.
    """
    intermediate_model = tf.keras.models.Model(
        model.inputs,model.get_layer(layer_name).output)
    activations = intermediate_model.predict(img_array)
    intermediate_model.summary()
    return [intermediate_model, activations] # Retourne le modèle et ces filtres

# Fonction pour visualiser les filtres sortant d'une couche du modèle ZFNet
def visualize_filters(activations, num_filters,name):
    """
    Visualise les filtres sortant d'une couche du modèle ZFNet.

    Args:
        activations (numpy.ndarray): Les activations de la couche.
        num_filters (int): Le nombre de filtres à visualiser.
        name (str): Le nom du fichier pour sauvegarder l'image.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, num_filters, figsize=(20, 20))
    for i in range(num_filters):
        ax = axes[i]
        ax.imshow(activations[0, :, :, i], cmap='viridis')
        ax.axis('off')
    plt.savefig(name)
    plt.show()

# Fonction pour évaluer le modèle sur une image en plus jolie
def evaluate_model_on_image(img, description, model):
    """
    Évalue le modèle sur une image et affiche la prédiction.

    Args:
        img (str): Le chemin de l'image à évaluer.
        description (str): Une description de l'image.
        model (tensorflow.keras.Model): Le modèle ZFNet.

    Returns:
        numpy.ndarray: La prédiction du modèle.
    """
    img_array = load_and_preprocess_image(img, target_size=(224, 224))
    #plt.imshow(img_array[0])
    prediction = model.predict(img_array)
    print(f"Prediction for {description}: {prediction}")
    return prediction

def plot_deconvoluted_stacks(model, img_array,names, num_stack=5):
    """
    Visualise les stacks déconvolués du modèle ZFNet.

    Args:
        model (tensorflow.keras.Model): Le modèle ZFNet.
        img_array (numpy.ndarray): L'image d'entrée sous forme de tableau numpy.
        num_stack (int): Le stack à visualiser.

    Returns:
        None
    """
    weights = model.get_weights()
    flipped_weights = [np.flip(w, axis=(0, 1)) for w in weights if len(w.shape) == 4]  # Inverser les filtres

    for nstack in range(1, num_stack + 1):
        activations = get_activations(model, img_array, names[nstack - 1])  # Obtenir les activations de la n-ième couche
        input_shape_deconv = activations[0].output.shape[1:]
        deconve = create_deconv_model(input_shape_deconv, stack=nstack)  # dimensions en sortie du premier stack
        deconve.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Set the weights for all layers in deconve
        flipped_weights_index = nstack - 1
        for layer in deconve.layers:
            if isinstance(layer, Conv2DTranspose):
                layer.set_weights([flipped_weights[flipped_weights_index], np.zeros(layer.filters)])
                flipped_weights_index -= 1

        i = deconve.predict(activations[1])

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img_array[0])
        plt.title(f'Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(i[0])
        plt.title(f'Deconvoluted Image - Stack {nstack}')

        plt.savefig(f'deconved_stack{nstack}')
        plt.show()

def apply_occlusion(img_array, x, y, occlusion_size=50):
    """
    Applique une occlusion sur une partie de l'image.

    Args:
        img_array (numpy.ndarray): L'image d'entrée sous forme de tableau numpy.
        x (int): La coordonnée x du coin supérieur gauche de la zone d'occlusion.
        y (int): La coordonnée y du coin supérieur gauche de la zone d'occlusion.
        occlusion_size (int, optional): La taille de la zone d'occlusion (carrée). Par défaut à 50.

    Returns:
        numpy.ndarray: L'image avec la zone d'occlusion appliquée.
    """
    occluded_img = img_array.copy()
    occluded_img[0, y:y+occlusion_size, x:x+occlusion_size, :] = 0
    return occluded_img
