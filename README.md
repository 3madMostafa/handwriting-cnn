# Handwritten Letter Recognition using CNN

## Overview

![Demo](https://nanonets.com/blog/content/images/2020/08/Hero-Gif-1.gif)

This project implements a Convolutional Neural Network (CNN) to recognize handwritten letters (A-Z) using the EMNIST Letters dataset. The model achieves letter classification through a deep learning architecture built with TensorFlow and Keras.

## Dataset

- **Source**: EMNIST Letters dataset
- **Format**: CSV file (`emnist-letters-train.csv`)
- **Image Size**: 28x28 pixels (grayscale)
- **Classes**: 26 (A-Z letters)
- **Split**: 80% training, 20% testing

## Model Architecture

### CNN Structure

```
Layer (type)                Output Shape              Parameters
=================================================================
Conv2D (32 filters, 3x3)    (None, 26, 26, 32)       320
MaxPooling2D (2x2)          (None, 13, 13, 32)       0
Conv2D (64 filters, 3x3)    (None, 13, 13, 64)       18,496
MaxPooling2D (2x2)          (None, 6, 6, 64)         0
Conv2D (128 filters, 3x3)   (None, 4, 4, 128)        73,856
MaxPooling2D (2x2)          (None, 2, 2, 128)        0
Flatten                     (None, 512)              0
Dense (64 units)            (None, 64)               32,832
Dense (128 units)           (None, 128)              8,320
Dense (26 units, softmax)   (None, 26)               3,354
=================================================================
```

### Key Features

- **3 Convolutional Layers**: Extract spatial features from images
- **3 Max Pooling Layers**: Reduce dimensionality and computational cost
- **2 Dense Layers**: Learn complex patterns (64 and 128 units)
- **Output Layer**: 26 units with softmax activation for letter classification
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Categorical crossentropy
- **Early Stopping**: Monitors validation loss with patience of 2 epochs

## Installation

### Requirements

```bash
pip install tensorflow
pip install pandas
pip install numpy
pip install scikit-learn
```

## Usage

### Training the Model

```python
# Load and prepare data
data = pd.read_csv("emnist-letters-train.csv").astype('float32')
X = data.drop('0', axis=1)
y = data['0']

# Split and reshape
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
train_x = np.reshape(train_x.values, (train_x.shape[0], 28, 28))
test_x = np.reshape(test_x.values, (test_x.shape[0], 28, 28))

# Train model
history = model.fit(train_X, train_y, epochs=1, 
                   callbacks=[early_stop],  
                   validation_data=(test_X, test_y))

# Save model
model.save('model_hand.h5')
```

### Making Predictions

```python
# Load the saved model
from tensorflow.keras.models import load_model
model = load_model('model_hand.h5')

# Make predictions
predictions = model.predict(test_X[:9])

# Convert predictions to letters
word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 
             6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 
             12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 
             18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 
             24: 'Y', 25: 'Z'}

predicted_letters = [word_dict[np.argmax(pred)] for pred in predictions]
```

## Project Structure

```
├── emnist-letters-train.csv    # Training dataset
├── train_model.py              # Main training script
├── model_hand.h5              # Saved trained model
└── README.md                  # This file
```

## Data Preprocessing

1. **Loading**: Read CSV file and convert to float32
2. **Splitting**: Separate features (X) and labels (y)
3. **Train-Test Split**: 80-20 split
4. **Reshaping**: Convert flat arrays to 28x28 images
5. **Channel Addition**: Add channel dimension for CNN (28, 28, 1)
6. **Label Encoding**: Convert labels to one-hot encoded vectors (26 classes)

## Training Configuration

- **Epochs**: 1 (configurable)
- **Optimizer**: Adam with learning rate 0.001
- **Batch Size**: Default (32)
- **Early Stopping**: 
  - Monitor: validation loss
  - Patience: 2 epochs
  - Mode: auto

## Model Performance

The model uses categorical crossentropy loss and accuracy metrics to evaluate performance on the validation set during training.

## Future Improvements

- Increase number of training epochs for better accuracy
- Add data augmentation (rotation, scaling, shifting)
- Implement dropout layers to prevent overfitting
- Experiment with different architectures (ResNet, VGG)
- Add batch normalization for faster convergence
- Implement learning rate scheduling
- Create a GUI for real-time handwriting recognition

## Dependencies

- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Scikit-learn**: Data splitting and preprocessing

## Notes

- Input images must be 28x28 pixels, grayscale
- The model expects normalized pixel values (0-255 range as float32)
- Labels are 0-indexed (0=A, 1=B, ..., 25=Z)
- The model is saved in HDF5 format

## License

This project is available for educational and research purposes.

## References

- EMNIST Dataset: Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters
- Keras Documentation: https://keras.io/
- TensorFlow Documentation: https://www.tensorflow.org/

---

**Note**: Adjust the number of epochs and consider adding more preprocessing steps for production use.
