# https://www.youtube.com/watch?v=LsdxvjLWkIY&t=208s

# In[1]:
import numpy as np
import cv2
import PIL.Image as Image
import os
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# In[2]
import pathlib 
data_dir = pathlib.Path('C:\\Users\\ASCan\\PycharmProjects\\Vision_Eindopdracht')

# In[2]:

# Stap 1: zoek uit welke imagesize het NN verwacht.
# dat blijkt in dit geval: 224x224.
IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape=IMAGE_SHAPE+(3,))])

# In[3]:

# Stap 2, laten we eens een plaatje met die size classifien door het NN.
gold_fish = Image.open("train/orange_19.jpg").resize(IMAGE_SHAPE)
gold_fish

# In[6]:

# Stap 2b, eerst moeten de pixels worden omgezet naar float waarden tussen 0 en 1
gold_fish = np.array(gold_fish)/255.0
gold_fish.shape

# In[7]:

# even wrappen in een extra dimensie ter grootte 1
# classifier.predict verwacht namelijk een array van images om te predicten
# dat array vullen we nu met 1 image.
(gold_fish[np.newaxis, ...]).shape

# In[8]:

# Met de volgende predict kunnen we makkelijk zien wat de shape van de output layer is.
result = classifier.predict(gold_fish[np.newaxis, ...])
result.shape

# In[8b]:

# Met de model summary had je dat ook kunnen zien:
classifier.summary()

# vreemd genoeg zien we hier geen details, zoals de vereiste input shape

# In[8b]:

# Zo kun je ook komen achter de verwachte input shape en de output shape
hub_layer = classifier.layers[0]

print(hub_layer.input_shape)
print(hub_layer.output_shape)

# In[9]:

# Het betreft een rij one-hot encoded outputs, een output per candidaat-klasse
# via argmax vinden we de hoogst scorende output.
predicted_label_index = np.argmax(result)
print(predicted_label_index)

# In[10]:

# via de index van de hoogstscorende output kunnen we opzoeken
# welk
# tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
image_labels = []
with open("ImageNetLabels.txt", "r") as f:
    image_labels = f.read().splitlines()
image_labels[predicted_label_index]

# In[20]:

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, cache_dir='.', untar=True)
# cache_dir indicates where to download data. I specified . which means current directory
# untar true will unzip it

# In[80]:
data_dir

# In[81]:

import pathlib
data_dir = pathlib.Path(data_dir)
data_dir

# In[82]:

list(data_dir.glob('*/*.jpg'))[:5]

# In[83]:

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# In[84]:

roses = list(data_dir.glob('roses/*'))
roses[:5]

# In[85]:

Image.open(str(roses[1]))

# In[86]:

tulips = list(data_dir.glob('tulips/*'))
Image.open(str(tulips[0]))

# In[87]:
# er zijn iets van 600 fotos per categorie
# met beperken tot 50 fotos per catetorie: list(data_dir.glob('roses/*'))[:50] -> acc = 84% ipv 89%
# met beperken tot 10 fotos per catetorie: list(data_dir.glob('roses/*'))[:50] -> acc = 77%(met data-augmentatie) ipv 89%

flowers_images_dict = {
    'roses': list(data_dir.glob('roses/*'))[:10],
    'daisy': list(data_dir.glob('daisy/*'))[:10],
    'dandelion': list(data_dir.glob('dandelion/*'))[:10],
    'sunflowers': list(data_dir.glob('sunflowers/*'))[:10],
    'tulips': list(data_dir.glob('tulips/*'))[:10]
}

# In[88]:

flowers_labels_dict = {
    'roses': 0,
    'daisy': 1,
    'dandelion': 2,
    'sunflowers': 3,
    'tulips': 4,
}

# In[89]:

flowers_images_dict['roses'][:5]

# In[90]:

str(flowers_images_dict['roses'][0])

# In[91]:

img = cv2.imread(str(flowers_images_dict['roses'][0]))

# In[92]:

img.shape

# In[93]:

cv2.resize(img, (224, 224)).shape

# In[94]:

X, y = [], []
for flower_name, images in flowers_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img,(224,224))
        X.append(resized_img)
        y.append(flowers_labels_dict[flower_name])

# In[95]:

X = np.array(X)
y = np.array(y)

# In[96]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# In[97]:

X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

# **Make prediction using pre-trained model on new flowers dataset**

# In[41]:

X[0].shape

# In[42]:

IMAGE_SHAPE+(3,)

# In[60]:

# mav: waarom doet hij dit? bij [In94] hierboven is het al geresized?
x0_resized = cv2.resize(X[0], IMAGE_SHAPE)
x1_resized = cv2.resize(X[1], IMAGE_SHAPE)
x2_resized = cv2.resize(X[2], IMAGE_SHAPE)

# In[61]:

plt.axis('off')
plt.imshow(X[0])

# In[63]:

plt.axis('off')
plt.imshow(X[1])

# In[64]:

plt.axis('off')
plt.imshow(X[2])

# In[72]:

predicted = classifier.predict(np.array([x0_resized, x1_resized, x2_resized]))
predicted = np.argmax(predicted, axis=1)
print(predicted)

# In[73]:

image_labels[795]

# <h3 style='color:purple'>Now take pre-trained model and retrain it using flowers images</h3>

# In[75]:

feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
pretrained_model_without_top_layer = hub.KerasLayer(feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

# In[98]:

# ik maak er even een classifier van om de shapes te kunnen inspecteren
tmpClassifier = tf.keras.Sequential([pretrained_model_without_top_layer])

tmp_hub_layer = tmpClassifier.layers[0]

# interessant, de feature layer heeft dus 1280 neuronen, iets meer dan de 1001 neuronen van de oorspronkelijke output layer.
print(tmp_hub_layer.input_shape)
print(tmp_hub_layer.output_shape)

# In[98]:

# simple dense layer results in test accuracy of about 86%

num_of_flowers = 5

model = tf.keras.Sequential([pretrained_model_without_top_layer,
                             tf.keras.layers.Dense(num_of_flowers) # by default, activation=None, so "logits" output.
                            ]) # Dense implements the operation: output = activation(dot(input,kernel) + bias)

model.summary()

# In[98]:

# Het bovenstaande netwerk werkt snel en goed.

# Nog even wat proberen de test accuracy te vergroten door overfitting te reduceren,
# door in plaats daarvan het volgende grotere, tragere netwerk te gebruiken:

# simple dense layer results in test accuracy of about 86%
# by using two of them, and some overfitting measures, it becomes 89%.

num_of_flowers = 5
model = tf.keras.Sequential([pretrained_model_without_top_layer,
                             tf.keras.layers.GaussianNoise(0.2),
                             tf.keras.layers.Dropout(0.2),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Dense(3000,activation='relu'),  # by default,activation='sigmoid', so "logits" output.
                             tf.keras.layers.GaussianNoise(0.2),
                             tf.keras.layers.Dropout(0.2),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Dense(num_of_flowers)  # by default, activation=None, so "logits" output.
                            ])  # Dense implements the operation: output = activation(dot(input, kernel) + bias)

model.summary()

# In[99]:

# SparseCategoricalCrossentropy is used when class labels are integers.
# CategoricalCrossentropy is used when class labels are one-hot vectors.
# In machine learning, the term "logits" refers to the output of a neural
# network layer that has no activation function applied to it.
# The term is most commonly used in the context of classification problems,
# where the neural network predicts the probability of each class label.

model.build() # initialises the weights. Not needed when using model.fit.

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['acc'])

model.fit(X_train_scaled, y_train, epochs=200)

# commented out, because we’re gonna train with augmentation instead, below:

model.evaluate(X_test_scaled, y_test)

# In[100]:

# Bovenstaand werkt gewoon. Voor de sport nog even verder mongeren om te kijken
# of het beter kan:

# let's see if we can do better adding using data augmentation:
# ah well.. it increased test accuracy from 0.888 to 0.892
# to achieve that, it took at least 100 times as long.

# important note: model.train_on_batch updates the model.
# so if you call model.fit first, it will continue training with
# the weights that resulted from model.fit.

# I don't know - perhaps it's better to first train with the quick method
# above, and after that improve with data augmentation below.
# Or perhaps if you first train with model.fit (with many epochs),
# the network already has reached a local minimum?
# Perhaps then, it may be better to first train with model.fit for 1 epoch?
# enough to randomize the weights and train 1 epoch. Or just doe that,
# using model.build().

# Okay, I tested it, and the results seem to be are better if first
# training with model.fit, and after that proceeding with the method below.

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the ImageDataGenerator with specified augmentations
datagen = ImageDataGenerator(
    rotation_range=20,       # Randomly rotate images by 20 degrees
    width_shift_range=0.1,   # Randomly shift images horizontally by 10%
    height_shift_range=0.1,  # Randomly shift images vertically by 10%
    shear_range=0.2,         # Randomly apply shear transformation with intensity of 0.2
    zoom_range=0.2,          # Randomly zoom in images by 20%
    horizontal_flip=True,    # Randomly flip images horizontally
    vertical_flip=True       # Randomly flip images horizontally
)

# Fit the ImageDataGenerator on the training data
datagen.fit(X_train_scaled)

# Create a generator that yields augmented data in batches during training
# batches larger than 512 do not fit in memory with current image size.
batch_size = min(len(X_train_scaled),512) # theres 2752 train members. 2752 is divisable by 64.
train_generator = datagen.flow(X_train_scaled, y_train, batch_size=batch_size)

# Train the model using the augmented data
num_epochs = 200
steps_per_epoch = len(X_train_scaled) // batch_size # Number of batches per epoch

try:
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for step in range(steps_per_epoch):
            # Get a batch of augmented images and labels from the generator
            x_batch, y_batch = train_generator.next()
            # Train the model on the batch of augmented data
            loss_value,_ = model.train_on_batch(x_batch, y_batch)
            print("Batch %d: loss = %.4f" % (step, loss_value))
            #print("Model weights = ", model.get_weights())
except:
    print('aborted')

model.evaluate(X_test_scaled, y_test)

# In[3]:
# Import all the images into a dict and make a fitting dict with labels.
Fruit_Images_Dict_Train = {
    'apples' : list(data_dir.glob('train/apple*.jpg')),
    'oranges': list(data_dir.glob('train/orange*.jpg')),
    'bananas': list(data_dir.glob('train/banana*.jpg')) 
}

Fruit_Images_Dict_Test = {
    'apples' : list(data_dir.glob('test/apple*.jpg')),
    'oranges': list(data_dir.glob('test/orange*.jpg')),
    'bananas': list(data_dir.glob('test/banana*.jpg')) 
}

Fruit_Labels_Dict = {
    'apples' : 0,
    'oranges': 1,
    'bananas': 2,
}

# In[4]:
# Re-size all the images to the correct shape (224, 224), add them to np.array and also make a scaled copy.
IMAGE_SHAPE = (224, 224)

X_Train, Y_Train = [], []
for fruit_types, images in Fruit_Images_Dict_Train.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img, IMAGE_SHAPE)
        X_Train.append(resized_img)
        Y_Train.append(Fruit_Labels_Dict[fruit_types])

X_Train = np.array(X_Train)
Y_Train = np.array(Y_Train)

X_Train_Scaled = X_Train / 255

X_Test, Y_Test = [], []
for fruit_types, images in Fruit_Images_Dict_Test.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img, IMAGE_SHAPE)
        X_Test.append(resized_img)
        Y_Test.append(Fruit_Labels_Dict[fruit_types])

X_Test = np.array(X_Test)
Y_Test = np.array(Y_Test)

X_Test_Scaled = X_Test / 255

# In[5]:
# Freeze the provided pretrained-model and re-train it. 
model = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape=IMAGE_SHAPE+(3,), trainable=False),
                             tf.keras.layers.GaussianNoise(0.2),
                             tf.keras.layers.Dropout(0.2),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Dense(3000, activation='relu'),
                             tf.keras.layers.GaussianNoise(0.2),
                             tf.keras.layers.Dropout(0.2),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Dense(3) # Amount of Itemclasses.
])  

model.compile(  
    optimizer="adam",            
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['acc']
)

model.fit(X_Train_Scaled, Y_Train, epochs=50)

model.evaluate(X_Test_Scaled, Y_Test)

# In[6]:

test1 = cv2.imread(str('test/orange_84.jpg'))
resized_img_test = cv2.resize(test1, IMAGE_SHAPE)

predictions = model.predict(X_Test_Scaled)

print(predictions)
# %%
