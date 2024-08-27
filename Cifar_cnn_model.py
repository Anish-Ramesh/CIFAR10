from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import numpy as np


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values


class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='valid'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(256, (2, 2), activation='relu', strides=(1, 1), padding='valid'),
    layers.Flatten(),
    
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))


test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')
print(test_loss)

def predict_class(image):
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])
    return class_names[predicted_class]

sample_image = x_test[6563]
predicted_label = predict_class(sample_image)
print(f'Predicted label: {predicted_label}')
model.save('cifar10_model.h5')
