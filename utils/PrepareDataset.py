import os
import shutil


def move_images():
    main_dir = input("Introduce your main folder (Example: 'output'):\n")

    sub_dir = input("Introduce origin and destiny"
                    " subfolder (Example: 'Grape/Grape_healthy'):\n")

    source_dir = os.path.join(main_dir, sub_dir)
    dest_dir = sub_dir

    if not os.path.exists(source_dir):
        print(f"El directorio de origen {source_dir} no existe.")
        return

    os.makedirs(dest_dir, exist_ok=True)

    moved_count = 0

    for filename in os.listdir(source_dir):
        if filename.endswith('.JPG') and 'Original' not in filename:
            source_file = os.path.join(source_dir, filename)
            dest_file = os.path.join(dest_dir, filename)

            shutil.move(source_file, dest_file)
            print(f"Moved: {filename}")
            moved_count += 1

    print(f"{moved_count} files moved")


if __name__ == "__main__":
    move_images()
