# %%
# %pip install numpy
# %pip install pandas
# %pip install opencv-python
# %pip install matplotlib
# %pip install plotly
# %pip install tensorflow
# %pip install setuptools
# %pip install keras
# %pip install scikit-learn
# %pip install mtcnn
# %pip install keras-facenet
# %pip install wandb

# %%
# Libraries

import tensorflow as tf

# Main
import os
import glob
# import gc
# import numpy as np
import pandas as pd
# # import cv2
# import time
# import random
# import datetime

# import wandb
# from wandb.keras import WandbCallback

# Visualization
# import matplotlib
import matplotlib.pyplot as plt
# import plotly
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots

# # Deep Learning
import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image as k_image
# from tensorflow.keras_facenet import FaceNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Lambda, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2, ResNet50, InceptionV3, VGG16
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import ReduceLROnPlateau

# print(tf.config.list_physical_devices('GPU'))

# import ssl

# ssl._create_default_https_context = ssl._create_unverified_context

# # Warning
# import warnings
# warnings.filterwarnings("ignore")


# # %%


#%%
class CFG:
    batch_size = 8
    img_height = 160
    img_width = 160
    epoch = 50 # TODO: czy jest sens robic mniej lub więcej epoch

    @classmethod
    def set_epoch(cls, new_epoch):
        cls.epoch = new_epoch

# define global constants
NUMBER_OF_CLASSES = 100

# %%
def create_dataframe(dataset_path):
    list_path = []
    labels = []

    identities = os.listdir(dataset_path)
    for identity in identities:
        identity_path = os.path.join(dataset_path, identity, "*")
        image_files = glob.glob(identity_path)

        identity_label = [identity] * len(image_files)

        list_path.extend(image_files)
        labels.extend(identity_label)

    data = pd.DataFrame({
        "image_path": list_path,
        "identity": labels
    })

    return data


# %%
dataset_path = "dataset/P1-72-100"
data = create_dataframe(dataset_path)

def write_to_output(*args):
    with open("results/output.txt", "a") as f:
        output_string = ' '.join(str(arg) for arg in args)
        print(output_string, file=f)


# %%
fig, axs = plt.subplots(5, 5, figsize=(12, 10))


print(data.head())
print(data.shape)

# Plot Images. Show 5 different persons by identity
# for i, ax in enumerate(axs.flat):
#     identity = data["identity"].unique()[i]
#     image_path = data[data["identity"] == identity]["image_path"].values[0]
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     ax.imshow(image)
#     ax.axis("off")
#     ax.set_title(identity)



# %%
def split_data(data, test_size, random_state=2023):
    X_train, X_test, y_train, y_test = train_test_split(
        data["image_path"], data["identity"],
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=data["identity"]
    )
    data_train = pd.DataFrame({
        "image_path": X_train,
        "identity": y_train
    })
    data_test = pd.DataFrame({
        "image_path": X_test,
        "identity": y_test
    })

    print(data_test["identity"].value_counts())
    print(data_train["identity"].value_counts())

    return data_train, data_test


def split_data_random_size(data, min_test_size=0.1, max_test_size=0.9, random_state=2023):
    unique_identities = data["identity"].unique()
    data_train = pd.DataFrame(columns=["image_path", "identity"])
    data_test = pd.DataFrame(columns=["image_path", "identity"])

    for identity in unique_identities:
        identity_data = data[data["identity"] == identity]
        test_size = random.uniform(min_test_size, max_test_size)

        X_train, X_test, y_train, y_test = train_test_split(
            identity_data["image_path"],
            identity_data["identity"],
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )

        data_train = pd.concat([data_train, pd.DataFrame({"image_path": X_train, "identity": y_train})], ignore_index=True)
        data_test = pd.concat([data_test, pd.DataFrame({"image_path": X_test, "identity": y_test})], ignore_index=True)
    # use write_to_output to show summory of how big test size for equal identity
    write_to_output(data_test["identity"].value_counts())
    write_to_output(data_train["identity"].value_counts())

    print(data_test["identity"].value_counts())
    print(data_train["identity"].value_counts())
    return data_train, data_test

# # %%
def create_model(num_classes, model_name="MobileNetV2"):
    if model_name == "MobileNetV2":
        base_model = MobileNetV2(
            input_shape=(CFG.img_height, CFG.img_width, 3),
            include_top=False,
            weights="imagenet"
        )
    elif model_name == "ResNet50":
        base_model = ResNet50(
            input_shape=(CFG.img_height, CFG.img_width, 3),
            include_top=False,
            weights="imagenet"
        )
    elif model_name == "InceptionV3":
        base_model = InceptionV3(
            input_shape=(CFG.img_height, CFG.img_width, 3),
            include_top=False,
            weights="imagenet"
        )
    elif model_name == "VGG16":
        base_model = VGG16(
            input_shape=(CFG.img_height, CFG.img_width, 3),
            include_top=False,
            weights="imagenet"
        )
    else:
        raise ValueError("Invalid model name. Please choose from 'MobileNetV2', 'ResNet50', 'InceptionV3', or 'VGG16'.")

    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(512, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(base_model.input, output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"]) # mona spróbować AdamW, zamiast categorical_crossentropy użyć triplet loss

    return model

# function to create and return flow_from_dataframe
def create_flow_from_dataframe(data):
    datagen = ImageDataGenerator(rescale=1./255)

    generator = datagen.flow_from_dataframe(
        data,
        x_col="image_path",
        y_col="identity",
        target_size=(CFG.img_height, CFG.img_width),
        batch_size=CFG.batch_size,
        class_mode="categorical"  # Use "categorical" class mode for multiple classes
    )

    return generator

# Function to test the model
def train_and_test_model(model, model_name, data_train, data_test):

    train_generator = create_flow_from_dataframe(data_train)
    test_generator = create_flow_from_dataframe(data_test)

    start_fitting_time = datetime.datetime.now()
    model.fit(train_generator, epochs=CFG.epoch, verbose=1)
    stop_fitting_time = datetime.datetime.now()

    write_to_output("Fitting duration: ", stop_fitting_time - start_fitting_time)


    test_loss, test_accuracy = model.evaluate(test_generator)

    # Write output to file
    write_to_output("Number of epoch: ", CFG.epoch)
    write_to_output("Test Loss: ", test_loss)
    write_to_output("Test Accuracy: ", test_accuracy)
    write_to_output("=========================")





# %%
# base params
test_size=0.2
test_sizes = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
data_train, data_test = split_data(data, test_size)
# models = ["VGG16", "InceptionV3", "ResNet50", "MobileNetV2"]
models = ["MobileNetV2"]



# %%
CFG.set_epoch(1)
write_to_output("")
write_to_output("3. Fitting time")

for model_name in models:
    write_to_output("Model name: ", model_name)
    for test_size in test_sizes:
        data_train, data_test = split_data(data, test_size)
        model = create_model(NUMBER_OF_CLASSES, model_name)
        write_to_output("Test size: ", test_size)
        train_and_test_model(model, model_name, data_train, data_test)

#