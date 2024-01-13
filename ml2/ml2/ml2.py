
#from keras.datasets import mnist
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.utils import np_utils

## 1. Load MNIST dataset
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

## 2. Preprocess the data
## Flatten images to 1D vector, normalize
#num_pixels = X_train.shape[1] * X_train.shape[2]
#X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32') / 255
#X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32') / 255

## change the labels into an array 
#y_train = np_utils.to_categorical(y_train)
#y_test = np_utils.to_categorical(y_test)
#num_classes = y_test.shape[1]

## 3. Build the model
#model = Sequential()
#model.add(Dense(num_pixels, input_dim=num_pixels, activation='relu'))
#model.add(Dense(num_classes, activation='softmax'))

## 4. Compile model
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

## 5. Train the model
#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

## 6. Evaluate the model
#scores = model.evaluate(X_test, y_test, verbose=0)
#print("Baseline Error: %.2f%%" % (100-scores[1]*100))

## 7. Save the model
#model.save('mnist_model.h5')


import tkinter as tk
from keras.models import load_model
import numpy as np
from PIL import Image, ImageDraw, ImageOps


# Loading the trained model
model = load_model('mnist_model.h5')

def predict_digit(canvas):
    # Resize the canvas content to 28x28 and convert to grayscale
    canvas.postscript(file='canvas_temp.eps')
    img = Image.open('canvas_temp.eps')
    img = img.resize((28, 28), Image.LANCZOS).convert('L')

    # Invert the colors (as the model was trained on white digits on a black background)
    img = ImageOps.invert(img)

    # Normalize pixel values
    img_array = np.array(img).astype('float32') / 255
    img_array = img_array.reshape(1, 784)  # Reshape to match model's input

    # Predict the digit
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)

    # Display the prediction
    prediction_label.config(text=f"Prediction: {digit}")


def draw(event):
    x, y = event.x, event.y
    canvas.create_oval((x-10, y-10, x+10, y+10), fill='black')

def clear():
    canvas.delete("all")


# Set up the GUI
window = tk.Tk()
window.title("Digit Recognizer")
window.geometry("400x600")  # Adjust the size of the window

# Add a heading
heading_label = tk.Label(window, text="Handwritten Digit Recognizer", font=("Helvetica", 20, "bold"))
heading_label.pack(pady=10)  # Add some padding for better spacing

# Add a description
description_label = tk.Label(window, text="Draw a digit (0-9) and click Predict to see the result.", font=("Helvetica", 12))
description_label.pack(pady=5)

# Create a canvas for drawing
canvas = tk.Canvas(window, width=280, height=280, bg='white')
canvas.pack(pady=20)  # Add padding around the canvas

# Bind the drawing method to mouse drag event
canvas.bind("<B1-Motion>", draw)

# Add a button to predict the digit
predict_button = tk.Button(window, text="Predict", command=lambda: predict_digit(canvas), font=("Helvetica", 14))
predict_button.pack(pady=10)  # Add padding around the button

# Add a button to clear the canvas
clear_button = tk.Button(window, text="Clear", command=clear, font=("Helvetica", 14))
clear_button.pack()

# Add a label to display the prediction
prediction_label = tk.Label(window, text="Prediction: ", font=("Helvetica", 16))
prediction_label.pack(pady=20)  # Add padding around the label

window.mainloop()

