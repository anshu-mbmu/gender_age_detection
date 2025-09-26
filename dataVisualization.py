

# importing important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
from tqdm.notebook import tqdm
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.preprocessing.image import load_img



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





df = pd.DataFrame()
df['image'], df['age'], df['gender'] = image_paths, age_labels, gender_labels
df.head(5)




df.tail(5)





gender_dict = {0:'Male', 1:'Female'}




img = Image.open(df['image'][10])
plt.axis('off')
plt.imshow(img)




sns.distplot(df['age'])




sns.countplot(x = df['gender'], data=df)





sampled_df = df.sample(n=25, random_state=42)  # Set random_state for reproducibility


plt.figure(figsize=(15, 20))

for index, row in enumerate(sampled_df.itertuples()):
    # row is a namedtuple with index and the columns as fields
    file = row.image
    age = row.age
    genders = row.gender


    plt.subplot(5, 5, index + 1)
    img = load_img(file)  # Load image
    img = np.array(img)  # Convert to array for displaying
    plt.imshow(img)
    plt.title(f"Age: {age} Gender: {gender_dict[int(genders)]}")
    plt.axis('off')

plt.tight_layout()
plt.show()
