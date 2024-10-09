from base import *
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, UpSampling2D
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input

input_shape = (224, 224, 3)  # Format
num_classes = 2  # Nombre de classes

############################################################################################################ TRAINED MODEL TEST
model = tf.keras.models.load_model('10.keras')
model.summary()


# Charger et prétraiter une image
img_path = 'IMG_20241003_184123.jpg'
img_array = load_and_preprocess_image(img_path, target_size=(224, 224))
plt.imshow(img_array[0])

model.predict(img_array)  # Prédiction de l'image [chat, chien]

############################################################################################################ TRAINED MODEL TEST

names = [layer.name for layer in model.layers]
indices_to_remove = [0,2,8,9,10, 11, 12]
names = [name for idx, name in enumerate(names) if idx not in indices_to_remove]
layer_outputs = [model.get_layer(name).output for name in names]
names # Noms des couches a appeller pour chaque stack du modèle


####################################################################
nstack = 3 # Choix du stack a visualiser (1, 2, 3, 4, 5)          ##
####################################################################


activations = get_activations(model, img_array, names[nstack-1]) # Obtenir les activations de la n-ième couche, ici la première
activations[1].shape

num_filters = 5  # Nombre de filtres à visualiser
visualize_filters(activations[1], num_filters,'conved_stack1.jpg') # Visualiser les filtres convoluer (activations[1] -> tableau convolué)

activations[0].get_config() # (actiavtions[0] -> couche de convolution)

############################################################################################################ DECONVOLUTION TEST
# visualisation DECONVOLUTION TEST du premier stack -> OK

weights = model.get_weights()
flipped_weights = [np.flip(w, axis=(0, 1)) for w in weights if len(w.shape) == 4]  # Inverser les filtres pour toutes les couches de convolution
flipped_weights[4].shape
input_shape_deconv=activations[0].output.shape[1:]
activations[0].summary()
#Flipped confirmation factor the first convolutional layer
weights[0][:,:,0,0]
flipped_weights[0][:,:,0,0]
flipped_weights[2].shape
deconve1 = alt_create_deconv_model(input_shape_deconv,stack=nstack) # dimensions en sortie du premier stack
deconve1.summary()
deconve1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set the weights for all layers in deconve1
flipped_weights_index = nstack-1
flipped_weights[4].shape
for layer in deconve1.layers:
    if isinstance(layer, Conv2DTranspose):
        layer.set_weights([flipped_weights[flipped_weights_index], np.zeros(layer.filters)])
        flipped_weights_index -= 1

i=deconve1.predict(activations[1])
i/=np.max(i)
i.shape

plt.imshow(img_array[0,:,:,:])
visualize_filters(activations[1], num_filters,'conved_stack1.jpg')
visualize_filters(i,3,'deconved_stack1.jpg')

#plt.savefig('successfully_deconved_bunny.jpg')
plt.imshow(i[0,:,:,:])
############################################################################################################ DECONVOLUTION TEST

# CA MARCHE :D
# reste a généraliser
