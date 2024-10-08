import sys
import json
from utils.SplitDataset import check_arguments
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from utils.SplitDataset import create_subdirectories, split_images


def prepare_data(train_dir, validation_dir, batch_size=32):
    """
    Prepares the training and validation data generators.
    Normalization is applied (rescale=1./255).
    """
    print("Preparing data generators...")

    # ImageDataGenerator for normalization
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Generator for training
    print("Creating training data generator...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Generator for validation
    print("Creating validation data generator...")
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, validation_generator


def define_model(input_shape, num_classes=4):
    """
    Defines and compiles a Convolutional Neural Network (CNN) model.
    """
    print(f"Defining CNN model with input shape: {input_shape} "
          "and {num_classes} classes...")

    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=input_shape))
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
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(),
                  metrics=['accuracy'])

    print("Model defined and compiled successfully.")
    return model


def train_model(base_dir, model, train_generator, validation_generator,
                epochs=10, save_model_path='best_model.keras'):
    """
    Trains the model and saves the best model based on validation accuracy.
    """
    print(f"Starting training for {epochs} epochs...")

    class_indices = train_generator.class_indices
    with open(f'utils/{base_dir}_class_indices.json', 'w') as f:
        json.dump(class_indices, f)
    print("Class indices saved to class_indices.json")

    # Checkpoint to save the best model based on validation accuracy
    save_model_path = f"utils/{base_dir}_{save_model_path}"
    checkpoint = ModelCheckpoint(save_model_path, monitor='val_accuracy',
                                 save_best_only=True, verbose=1)

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=(
            validation_generator.samples // validation_generator.batch_size
        ),
        callbacks=[checkpoint]
    )

    print(f"Training completed. Model saved as {save_model_path}.")
    return history


def main():

    base_dir = check_arguments()

    str = input("Do you want to split the images into "
                "training and validation folders? (Y/N): \n")
    if str.lower() == 'y':
        print(
            f"Creating training and validation "
            f"folders from base directory: {base_dir}"
        )
        train_dir, validation_dir = create_subdirectories(base_dir)
        print(f"Splitting images into {train_dir} and {validation_dir}...")
        split_images(base_dir, train_dir, validation_dir)
        print(
            f"Images successfully split into {train_dir} and {validation_dir}."
        )

    elif str.lower() == 'n':
        train_dir = input("Enter the training directory name: \n")
        validation_dir = input("Enter the validation directory name: \n")

    else:
        print("Invalid input.")
        sys.exit(1)

    train_generator, validation_generator = prepare_data(train_dir,
                                                         validation_dir)

    image_shape = train_generator.image_shape
    model = define_model(input_shape=image_shape,
                         num_classes=train_generator.num_classes)

    print("Starting model training process...")
    train_model(base_dir, model, train_generator,
                validation_generator, epochs=10)

    print("Training process finished successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
