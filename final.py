import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the saved model
model = load_model('D:/Python Projects/AI & ML/Cats n Dogs/CatsAndDogsModel.h5')

# Load the image you want to classify
img_path = 'D:/Python Projects/AI & ML/Cats n Dogs/test_image.jpg'
img = image.load_img(img_path, target_size=(150, 150))

# Convert the image to a numpy array
img_array = image.img_to_array(img)

# Expand the dimensions of the image so it can be used as input to the model
img_array = np.expand_dims(img_array, axis=0)

# Scale the image pixels between 0 and 1
img_array /= 255.

# Use the model to predict the class of the image
prediction = model.predict(img_array)

print('\n','\n')

# Print the predicted class
if prediction == 0.5:
    print('A cat and a dog?')
elif prediction < 0.5:
    print('The image is a cat')
else:
    print('The image is a dog')

print('\n')
input("press enter to close")