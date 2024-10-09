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
img_path = 'data/test/Cat/1.jpg'
img_array = load_and_preprocess_image(img_path, target_size=(224, 224))
plt.imshow(img_array[0])

model.predict(img_array)  # Prédiction de l'image [chat, chien]

############################################################################################################ TRAINED MODEL TEST

names = [layer.name for layer in model.layers]
layer_outputs = [model.get_layer(name).output for name in names]
names # Noms des couches du modèle

activations = get_activations(model, img_array, names[1]) # Obtenir les activations de la n-ième couche, ici la première
activations[1].shape

num_filters = 5  # Nombre de filtres à visualiser
visualize_filters(activations[1], num_filters,'conved_stack1.jpg') # Visualiser les filtres convoluer (activations[1] -> tableau convolué)

activations[0].get_config() # (actiavtions[0] -> couche de convolution)

############################################################################################################ DECONVOLUTION TEST
# visualisation DECONVOLUTION TEST du premier stack -> OK

weights = model.get_weights()
weights[0].shape  # Poids de la première couche de convolution

flipped_weights = np.flip(weights[0], axis=(0, 1)) # le papier a dit que les filtres doivent être inversés

#Flipped confirmation
weights[0][:,:,0,0]
flipped_weights[:,:,0,0]


deconve1 = create_deconv_test((54, 54, 96)) # dimensions en sortie du premier stack
deconve1.summary()
deconve1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

deconve1.layers[2].set_weights([flipped_weights, np.zeros(3)]) # Set the weights, biases are set to 0

i=deconve1.predict(activations[1])
i.shape

plt.imshow(img_array[0,:,:,:])
visualize_filters(activations[1], num_filters,'conved_stack1.jpg')
visualize_filters(i,3,'deconved_stack1.jpg')

#plt.savefig('successfully_deconved_bunny.jpg')

plt.imshow(i[0,:,:,:])
############################################################################################################ DECONVOLUTION TEST

# CA MARCHE :D
# reste a généraliser
