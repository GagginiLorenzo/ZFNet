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

############################################################################################################ INIT

"""
Cette section initialise le modèle, charge une image, la prétraite et fait une prédiction.
Le score de prédiction est ensuite affiché avec l'image.
"""

input_shape = (224, 224, 3)  # Format
num_classes = 2  # Number of classes

# Load the pre-trained model
model = tf.keras.models.load_model('10.keras')
model.summary()

# Load and preprocess an image
img_path = 'exp/10115.jpg'
img_array = load_and_preprocess_image(img_path, target_size=(224, 224))
plt.imshow(img_array[0])

# Predict the class of the image [cat, dog]
prediction = model.predict(img_array)

# Display the image and its prediction score
plt.title(f'Prediction Score: {prediction[0][0]:.4f}')
plt.savefig(f'prediction')
plt.show()

############################################################################################################ TRAINED MODEL TEST

"""
Cette section visualise les activations d'une couche spécifique du modèle.
Elle filtre d'abord certaines couches en fonction de leurs indices, puis obtient les activations
de la couche spécifiée (nstack) et visualise un nombre défini de filtres à partir de ces activations.
"""

names = [layer.name for layer in model.layers]
indices_to_remove = [0, 2, 6, 8, 9, 10, 11, 12]
names = [name for idx, name in enumerate(names) if idx not in indices_to_remove]
layer_outputs = [model.get_layer(name).output for name in names]

####################################################################
nstack = 1  # Choix du stack a visualiser (1, 2, 3, 4, 5)          ##
####################################################################

activations = get_activations(model, img_array, names[nstack - 1])  # Obtenir les activations de la n-ième couche
activations[1].shape

num_filters = 5  # Nombre de filtres à visualiser
visualize_filters(activations[1], num_filters, 'conved_stack1.jpg')  # Visualiser les filtres convoluer

############################################################################################################ DECONVOLUTION TEST

"""
Cette section effectue un test de déconvolution sur le modèle.
Elle inverse les filtres du modèle, crée un modèle de déconvolution, définit les poids,
et visualise les activations et déconvolutions d'une couche spécifique.
"""

weights = model.get_weights()
flipped_weights = [np.flip(w, axis=(0, 1)) for w in weights if len(w.shape) == 4]  # Inverser les filtres
input_shape_deconv = activations[0].output.shape[1:]

deconve1 = create_deconv_model(input_shape_deconv, stack=nstack)  # Dimensions en sortie du stack
deconve1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set the weights for all layers in deconve1
flipped_weights_index = nstack - 1
for layer in deconve1.layers:
    if isinstance(layer, Conv2DTranspose):
        layer.set_weights([flipped_weights[flipped_weights_index], np.zeros(layer.filters)])
        flipped_weights_index -= 1

i = deconve1.predict(activations[1])

plt.imshow(img_array[0])
visualize_filters(activations[1], num_filters, 'conved_stack1.jpg')
visualize_filters(i, 3, 'deconved_stack1.jpg')
plt.imshow(i[0])

############################################################################################################ PLOT DECONVOLUTED STACKS
"""
Cette section visualise les déconvolutions de plusieurs couches du modèle.
"""

plot_deconvoluted_stacks(model, img_array,names, num_stack=5)

############################################################################################################ ROTATION TEST

from PIL import Image

image_paths = ['102.jpg', '10041.jpg', '10115.jpg']  # Ajoutez les chemins vers vos images ici

results = []

for img_path in image_paths:
    image = Image.open(img_path)
    for i in range(1, 380):
        rotated = image.rotate(i, expand=True)
        rotated.save(f'exp/rotated_{img_path}.jpg')
        rota = load_and_preprocess_image(f'exp/rotated_{img_path}.jpg', target_size=(224, 224))
        prediction_rotated = model.predict(rota)
        results.append((f'{img_path} rotated {i} degrees', prediction_rotated))

c = 0  # cat 0 dog 1

# Plot
plt.figure(figsize=(10, 6))

for img_path in image_paths:
    img_results = [result for result in results if img_path in result[0]]
    degrees = list(range(1, 380))
    cat_values = [result[1][0][c] for result in img_results]
    plt.plot(degrees, cat_values, label=f'Probability of True Class vs Degrees of Rotation for {img_path}')

plt.xlabel('Degrees of Rotation')
plt.ylabel('Probability of True Class')
plt.title('Probability of True Class vs Degrees of Rotation')
plt.legend()
plt.grid(True)
plt.savefig('rotation_test.png')
plt.show()

############################################################################################################ FLIP TEST

image_paths = ['102.jpg', '10041.jpg', '10115.jpg']  # Ajoutez les chemins vers vos images ici
results = []

for img_path in image_paths:
    image = Image.open(img_path)

    # Flip images horizontally and vertically
    flipped_horizontal = image.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_vertical = image.transpose(Image.FLIP_TOP_BOTTOM)
    flipped_horizontal.save(f'exp/flipped_horizontal_{img_path}')
    flipped_vertical.save(f'exp/flipped_vertical_{img_path}')

    # Evaluate the model on flipped images
    prediction_horizontal = evaluate_model_on_image(f'exp/flipped_horizontal_{img_path}', f'flipped_horizontal', model=model)
    prediction_vertical = evaluate_model_on_image(f'exp/flipped_vertical_{img_path}', f'flipped_vertical', model=model)

    # Extract the probabilities for plotting
    cat_value_horizontal = prediction_horizontal[0][c]
    cat_value_vertical = prediction_vertical[0][c]

    image_flipped_horizontal = load_and_preprocess_image(f'exp/flipped_horizontal_{img_path}', target_size=(224, 224))
    image_flipped_vertical = load_and_preprocess_image(f'exp/flipped_vertical_{img_path}', target_size=(224, 224))

    results.append((img_path, cat_value_horizontal, cat_value_vertical, image_flipped_horizontal, image_flipped_vertical))

# Plot
plt.figure(figsize=(12, 6 * len(image_paths)))

for idx, (img_path, cat_value_horizontal, cat_value_vertical, image_flipped_horizontal, image_flipped_vertical) in enumerate(results):
    plt.subplot(len(image_paths), 2, 2 * idx + 1)
    plt.imshow(image_flipped_horizontal[0])
    plt.title(f'Flipped Horizontal Prediction Score: {cat_value_horizontal:.4f}')

    plt.subplot(len(image_paths), 2, 2 * idx + 2)
    plt.imshow(image_flipped_vertical[0])
    plt.title(f'Flipped Vertical Prediction Score: {cat_value_vertical:.4f}')

plt.savefig('flipped_images_predictions.png')
plt.show()

############################################################################################################ CROP TEST

image_paths = ['102.jpg', '10041.jpg', '10115.jpg']  # Ajoutez les chemins vers vos images ici
results = []

for img_path in image_paths:
    image = Image.open(img_path)
    image_cropped = load_and_preprocess_image(f'{img_path.split(".")[0]}_cropped.jpg', target_size=(224, 224))
    image_base = load_and_preprocess_image(img_path, target_size=(224, 224))
    prediction_cropped = model.predict(image_cropped)
    prediction_base = model.predict(image_base)
    results.append((img_path, prediction_base, prediction_cropped))

plt.figure(figsize=(12, 6 * len(image_paths)))

for idx, (img_path, prediction_base, prediction_cropped) in enumerate(results):
    plt.subplot(len(image_paths), 2, 2 * idx + 1)
    image_base = load_and_preprocess_image(img_path, target_size=(224, 224))
    plt.imshow(image_base[0])
    plt.title(f'Base Image Prediction Score: {prediction_base[0][c]:.4f}')

    plt.subplot(len(image_paths), 2, 2 * idx + 2)
    image_cropped = load_and_preprocess_image(f'{img_path.split(".")[0]}_cropped.jpg', target_size=(224, 224))
    plt.imshow(image_cropped[0])
    plt.title(f'Cropped Image Prediction Score: {prediction_cropped[0][c]:.4f}')
plt.savefig('cropped_images_predictions.png')
plt.show()

############################################################################################################ SHIFT TEST

image_paths = ['102.jpg', '10041.jpg', '10115.jpg']  # Ajoutez les chemins vers vos images ici
results = []

for img_path in image_paths:
    image = Image.open(img_path)
    for i in range(-50, 51, 5):
        # Left-right translation
        translated_lr = image.transform(image.size, Image.AFFINE, (1, 0, i, 0, 1, 0))
        translated_lr.save(f'exp/translated_lr_{img_path.split(".")[0]}.jpg')
        trans_lr = load_and_preprocess_image(f'exp/translated_lr_{img_path.split(".")[0]}.jpg', target_size=(224, 224))
        prediction_translated_lr = model.predict(trans_lr)
        results.append((f'{img_path} translated left-right {i} pixels', prediction_translated_lr))

        # Up-down translation
        translated_ud = image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, i))
        translated_ud.save(f'exp/translated_ud_{img_path.split(".")[0]}.jpg')
        trans_ud = load_and_preprocess_image(f'exp/translated_ud_{img_path.split(".")[0]}.jpg', target_size=(224, 224))
        prediction_translated_ud = model.predict(trans_ud)
        results.append((f'{img_path} translated up-down {i} pixels', prediction_translated_ud))

c = 0 # chat 0, chien 1

translations = list(range(-50, 51, 5))

# Tracé pour la translation gauche-droite
plt.figure(figsize=(10, 6))
for img_path in image_paths:
    img_results_lr = [result for result in results if f'{img_path} translated left-right' in result[0]]
    cat_values_lr = [result[1][0][c] for result in img_results_lr]
    plt.plot(translations, cat_values_lr, label=f'{img_path} Probability of True Class vs Left-Right Translation')

plt.xlabel('Translation (pixels)')
plt.ylabel('Probability of True Class')
plt.title('Probability of True Class vs Left-Right Translation')
plt.legend()
plt.grid(True)
plt.savefig('left_right_translation.png')
plt.show()

# Tracé pour la translation haut-bas
plt.figure(figsize=(10, 6))
for img_path in image_paths:
    img_results_ud = [result for result in results if f'{img_path} translated up-down' in result[0]]
    cat_values_ud = [result[1][0][c] for result in img_results_ud]
    plt.plot(translations, cat_values_ud, label=f'{img_path} Probability of True Class vs Up-Down Translation')

plt.xlabel('Translation (pixels)')
plt.ylabel('Probability of True Class')
plt.title('Probability of True Class vs Up-Down Translation')
plt.legend()
plt.grid(True)
plt.savefig('up_down_translation.png')
plt.show()

############################################################################################################ SCALE TEST

image_paths = ['102.jpg', '10041.jpg', '10115.jpg']  # Ajoutez les chemins vers vos images ici
results = []

for img_path in image_paths:
    image = Image.open(img_path)
    for scale in np.arange(1.0, 2.01, 0.1):
        # Scale transformation
        scaled = image.resize((int(image.width * scale), int(image.height * scale)))
        scaled.save(f'exp/scaled_{img_path.split(".")[0]}.jpg')
        scaled_img = load_and_preprocess_image(f'exp/scaled_{img_path.split(".")[0]}.jpg', target_size=(224, 224))
        prediction_scaled = model.predict(scaled_img)
        results.append((f'{img_path} scaled {scale:.1f}x', prediction_scaled))

c = 0  # cat 0 dog 1

scales = np.arange(1.0, 2.01, 0.1)
cat_values_scale = [result[1][0][c] for result in results]

# Plotting
plt.figure(figsize=(10, 6))
for img_path in image_paths:
    img_results = [result for result in results if img_path in result[0]]
    cat_values_scale = [result[1][0][c] for result in img_results]
    plt.plot(scales, cat_values_scale, label=f'Probability of True Class vs Scale for {img_path}')

plt.xlabel('Scale Factor')
plt.ylabel('Probability of True Class')
plt.title('Probability of True Class vs Scale')
plt.legend()
plt.grid(True)
plt.savefig('scale.png')

############################################################################################################ FULL OCCULSION TEST

img_path = '102.jpg' # Ajoutez lee chemin vers votre image ici
img_array = load_and_preprocess_image(img_path, target_size=(224, 224))

occlusion_size = 50
step_size = 10
img_height, img_width, _ = img_array.shape[1:]

true_class = 0  # Changez ceci en fonction de votre classe réelle (0 pour chat, 1 pour chien dans notre cas)

# Initialiser les résultats
results = []

# Appliquer l'occlusion et prédire
for y in range(0, img_height, step_size):
    for x in range(0, img_width, step_size):
        occluded_img = apply_occlusion(img_array, x, y, occlusion_size)
        plt.imshow(occluded_img[0])
        prediction = model.predict(occluded_img)
        true_class_prob = prediction[0][true_class]
        results.append((x, y, true_class_prob))

# Convertir les résultats en un tableau numpy
results = np.array(results)
heatmap = np.zeros((img_height, img_width))

for (x, y, prob) in results:
    x, y = int(x), int(y)  # S'assurer que x et y sont des entiers
    heatmap[y:y+occlusion_size, x:x+occlusion_size] = prob

plt.figure(figsize=(20, 6))

plt.subplot(1, 2, 1)
plt.imshow(heatmap, cmap='viridis', interpolation='nearest', origin='upper')
plt.colorbar(label='Probability of True Class')
plt.xlabel('X Position of Occlusion (pixels)')
plt.ylabel('Y Position of Occlusion (pixels)')
plt.title('Probability of True Class vs Position of Occlusion')
plt.grid(False)

# Plot
plt.subplot(1, 2, 2)

plt.title('Original Image')
plt.imshow(img_array[0])

plt.savefig('occlusion_test.png')
plt.show()
