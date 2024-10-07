import os
import sys
from utils.SplitDataset import split_images, create_subdirectories, check_arguments
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


def prepare_data(train_dir, batch_size=32):
    """
    Prepares the training data generator without augmentation.
    Normalization is applied (rescale=1./255).
    """
    print("Preparing data generators...")

    # ImageDataGenerator for training (normalization only)
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # 80/20 split

    # Generator for training (80% of the data)
    print("Creating training data generator...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    # Generator for validation (20% of the data)
    print("Creating validation data generator...")
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator


def define_model(input_shape, num_classes=4):
    """
    Defines and compiles a Convolutional Neural Network (CNN) model.
    """
    print(f"Defining CNN model with input shape: {input_shape} and {num_classes} classes...")

    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten and Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))  # Number of classes

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    print("Model defined and compiled successfully.")
    return model


def train_model(model, train_generator, validation_generator, epochs=10, save_model_path='best_model.keras'):
    """
    Trains the model and saves the best model based on validation accuracy.
    """
    print(f"Starting training for {epochs} epochs...")

    # Checkpoint to save the best model based on validation accuracy
    checkpoint = ModelCheckpoint(save_model_path, monitor='val_accuracy', save_best_only=True, verbose=1)

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        callbacks=[checkpoint]
    )

    print(f"Training completed. Model saved as {save_model_path}.")
    return history


def main():
    # Step 1: Check the arguments
    print("Checking arguments...")
    base_dir = check_arguments()

    # Step 2: Create the training and validation folders
    """ print(f"Creating training and validation folders from base directory: {base_dir}")
    train_dir, validation_dir = create_subdirectories(base_dir)

    # Step 3: Split the images between training and validation
    print(f"Splitting images into {train_dir} and {validation_dir}...")
    split_images(base_dir, train_dir, validation_dir)
    print(f"Images successfully split into {train_dir} and {validation_dir}.")"""

    
    train_dir = 'Grape_train'
    validation_dir = 'Grape_validation'    
    
    
    # Step 4: Prepare the data
    train_generator, validation_generator = prepare_data(train_dir)

    # Step 5: Define the model
    image_shape = train_generator.image_shape
    model = define_model(input_shape=image_shape, num_classes=train_generator.num_classes)

    # Step 6: Train the model
    print("Starting model training process...")
    train_model(model, train_generator, validation_generator, epochs=10)

    print("Training process finished successfully.")


if __name__ == "__main__":
    main()
