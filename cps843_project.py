import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, MaxPooling3D, Flatten, Dense
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
import tensorflow as tf

"""### Function for extracting first 40 frames per video clip"""

def extract_frames(video_path, num_framer = 40, resize=(64,64)):
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read() # ret basically returns true if frame is successfully read
        if ret == False:
            break;
        frame = cv2.resize(frame, resize)
        frames.append(frame)
        frame_count = frame_count + 1

        if (frame_count == 40):
            break
    cap.release()
    return np.array(frames) # Return frames converted to np array for faster operations

"""### Actual Extraction of frames from video clips"""

data = [] # where our videos(video frames) are going
labels = [] # class labels
video_dir = "./data/"

for category in os.listdir(video_dir):
    category_path = os.path.join(video_dir, category)
    for video in os.listdir(category_path):
        video_path = os.path.join(category_path, video)
        frames = extract_frames(video_path)
        if(len(frames) == 40):
            data.append(frames)
            labels.append(category)

# Get numerical labels instead of regular labels
label_to_index = {label: i for i, label in enumerate(np.unique(labels))}
indexed_labels = [label_to_index[label] for label in labels]

"""### Splitting Data into Train/Test split + Normalizing Data"""

X_train, X_test, y_train, y_test = train_test_split(data, indexed_labels, test_size=0.2, random_state=42)


# Normalize data
X_train = np.array(X_train) / 255.0
X_test = np.array(X_test) / 255.0
y_train = np.array(y_train)
y_test = np.array(y_test)

# Reshape data for ConvLSTM
X_train = X_train.reshape((-1, 40, 64, 64, 3))  # 40 frames - 64 x 64 pic size and 3 color channels
X_test = X_test.reshape((-1, 40, 64, 64, 3))

"""### ConvLSTM Model Creation"""

model = Sequential([
    ConvLSTM2D(32, kernel_size=(3, 3), activation='relu',
               kernel_regularizer=regularizers.l2(0.01),  # Adding L2 regularization
               input_shape=(40, 64, 64, 3), return_sequences=True),
    BatchNormalization(),
    MaxPooling3D(pool_size=(1, 2, 2)),  # reduce spatial dimensions
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l1(0.01)),  # Adding L1 regularization
    # Adding Dropout for regularization
    tf.keras.layers.Dropout(0.5),  # Adjust dropout rate as needed
    Dense(len(np.unique(labels)), activation='softmax')
])

"""### Training Model"""

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop

# Custom learning rate put here
custom_optimizer = Adam(learning_rate=0.0001)

model.compile(optimizer=custom_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

"""### Evaluating Accuracy and Loss"""

evaluation = model.evaluate(X_test, y_test)

accuracy = evaluation[1]

print(f"Test Accuracy: {accuracy * 100:.2f}%")

train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'g', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

"""### Testing with new data"""

ja_morant_dunk = './test_clip.mp4'
steph_curry_three = './test_clip2.mp4'

# Get frames from vid +
# normalize +
# reshape for convlstm
test_frames = extract_frames(ja_morant_dunk)
test_frames = np.array(test_frames) / 255.0
test_frames = test_frames.reshape((-1, 40, 64, 64, 3))

# make prediction
predictions = model.predict(test_frames)
predicted_class_index = np.argmax(predictions)

index_to_label = {v: k for k, v in label_to_index.items()}

# just getting class here
predicted_class = index_to_label[predicted_class_index]
print(f"The predicted class of the test video is: {predicted_class}")

"""### Action Counter Function"""

def count_actions_in_video(video_path, window_size=40, overlap=20):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    segments = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        segments.append(frame)

    cap.release()

    # process frames
    num_segments = len(segments) // overlap
    action_counts = {"Shots": 0, "Dunks": 0, "FT": 0, "Other": 0}

    for i in range(0, len(segments), overlap):
        end_idx = i + window_size

        # just checking to see if the segment has enough frames
        if end_idx <= len(segments):
            segment_frames = segments[i:end_idx]

            # Classify the frame into 1 of the classes
            segment_frames = [cv2.resize(frame, (64, 64)) for frame in segment_frames]
            segment_frames = np.array(segment_frames) / 255.0
            segment_frames = segment_frames.reshape((-1, window_size, 64, 64, 3))

            predictions = model.predict(segment_frames)
            predicted_class_index = np.argmax(predictions)

            index_to_label = {v: k for k, v in label_to_index.items()}
            predicted_class = index_to_label[predicted_class_index]

            # Increment action counts
            if predicted_class == "Shots":
                action_counts["Shots"] += 1
            elif predicted_class == "Dunks":
                action_counts["Dunks"] += 1

    return action_counts

# Testing the action counter here
video_path = './track_test_clip2.mp4'
counts = count_actions_in_video(video_path)
print(counts)
