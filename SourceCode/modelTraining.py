

# importing important libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm.notebook import tqdm
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

from keras.preprocessing.image import load_img
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input



from tqdm import tqdm
import os

# Base directory of the dataset
BASE_DIR = "Dataset/UTKFace"

# Lists to store extracted data
image_paths = []
age_labels = []
gender_labels = []

# Process each file in the dataset
for filename in tqdm(os.listdir(BASE_DIR)):
    image_path = os.path.join(BASE_DIR, filename)
    temp = filename.split('_')  # Split filename into components
    age = int(temp[0])  # Extract age
    gender = int(temp[1])  # Extract gender
    image_paths.append(image_path)  # Add image path
    age_labels.append(age)  # Add age label
    gender_labels.append(gender)  # Add gender label




# convert to dataframe
df = pd.DataFrame()
df['image'], df['age'], df['gender'] = image_paths, age_labels, gender_labels
df.head(5)




df.tail(5)




# map labels for gender
gender_dict = {0:'Male', 1:'Female'}




from PIL import Image #Python Imaging Library (Pillow) module for image processing tasks such as resizing and handling grayscale images.
from keras.preprocessing.image import load_img
import numpy as np
from tqdm import tqdm #It provides a progress bar for iterating through a loop. This is useful for visual feedback when processing a large number of images.


def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, color_mode='grayscale')
        img = img.resize((128, 128), Image.Resampling.LANCZOS)  # LANCZOS is a high-quality down-sampling filter
        img = np.array(img)
        features.append(img)

    features = np.array(features)
    # For grayscale images, reshape to (num_samples, 128, 128, 1)
    features = features.reshape(len(features), 128, 128, 1)  # 1 channel for grayscale
    return features




X = extract_features(df['image'])



X.shape




# normalize the images
X = X/255.0




y_gender = np.array(df['gender'])
y_age = np.array(df['age'])




input_shape = (128, 128, 1) #dimensions of data fed into network




inputs = Input((input_shape))
# convolutional layers
conv_1 = Conv2D(32, kernel_size=(3, 3), activation='relu') (inputs) #32 filters of 3x3 size - ReLu activation function
maxp_1 = MaxPooling2D(pool_size=(2, 2)) (conv_1) #Reduces spatial dimensions by taking the maximum value in each 2x2 region.
conv_2 = Conv2D(64, kernel_size=(3, 3), activation='relu') (maxp_1)
maxp_2 = MaxPooling2D(pool_size=(2, 2)) (conv_2)
conv_3 = Conv2D(128, kernel_size=(3, 3), activation='relu') (maxp_2)
maxp_3 = MaxPooling2D(pool_size=(2, 2)) (conv_3)
conv_4 = Conv2D(256, kernel_size=(3, 3), activation='relu') (maxp_3)
maxp_4 = MaxPooling2D(pool_size=(2, 2)) (conv_4)

flatten = Flatten() (maxp_4) #Converts the 2D feature maps (output from the final pooling layer) into a 1D vector.

# fully connected layers
dense_1 = Dense(256, activation='relu') (flatten)
dense_2 = Dense(256, activation='relu') (flatten)

dropout_1 = Dropout(0.3) (dense_1)
dropout_2 = Dropout(0.3) (dense_2)

output_1 = Dense(1, activation='sigmoid', name='gender_out') (dropout_1)
output_2 = Dense(1, activation='relu', name='age_out') (dropout_2)

model = Model(inputs=[inputs], outputs=[output_1, output_2])

model.compile(loss=['binary_crossentropy', 'mae'], optimizer='adam', metrics=['accuracy', 'mae'])

# This CNN model extracts features from images and predicts both gender (classification) and age (regression).
# It uses shared convolutional layers for feature extraction and separate dense layers for each task.



model.summary()




# train model
history = model.fit(x=X, y=[y_gender, y_age], batch_size=32, epochs=30, validation_split=0.2)




# Save the trained model to a file
# This saves the model's architecture, weights, and the state of the optimizer.
model.save('Models/age_gender_model.h5')

print("Model saved successfully to 'age_gender_model.h5'")




# Alternatively, save to the recommended .keras format
model.save('Models/age_gender_model.keras')
print("Model saved successfully to 'age_gender_model.keras'")




# Print the final accuracy for gender and age predictions
gender_accuracy = history.history['gender_out_accuracy'][-1]
age_accuracy = history.history['age_out_mae'][-1]

print(f"Final Gender Prediction Accuracy: {gender_accuracy:.4f}")
print(f"Final Age Prediction MAE: {age_accuracy:.4f}")




# plot results for gender
acc = history.history['gender_out_accuracy']
val_acc = history.history['val_gender_out_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.figure()

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss Graph')
plt.legend()
plt.show()




# plot results for age
loss = history.history['age_out_mae']
val_loss = history.history['val_age_out_mae']
epochs = range(len(loss))

plt.plot(epochs, loss, 'b', label='Training MAE')
plt.plot(epochs, val_loss, 'r', label='Validation MAE')
plt.title('MAE Graph')
plt.legend()
plt.show()




image_index = 100
print("Original Gender:", gender_dict[y_gender[image_index]], "Original Age:", y_age[image_index])
# predict from model
pred = model.predict(X[image_index].reshape(1, 128, 128, 1))
pred_gender = gender_dict[round(pred[0][0][0])]
pred_age = round(pred[1][0][0])
print("Predicted Gender:", pred_gender, "Predicted Age:", pred_age)
plt.axis('off')
plt.imshow(X[image_index].reshape(128, 128), cmap='gray')




image_index = 2413
print("Original Gender:", gender_dict[y_gender[image_index]], "Original Age:", y_age[image_index])
# predict from model
pred = model.predict(X[image_index].reshape(1, 128, 128, 1))
pred_gender = gender_dict[round(pred[0][0][0])]
pred_age = round(pred[1][0][0])
print("Predicted Gender:", pred_gender, "Predicted Age:", pred_age)
plt.axis('off')
plt.imshow(X[image_index].reshape(128, 128), cmap='gray')






