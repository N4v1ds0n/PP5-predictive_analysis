import os


def remove_non_image_files(data_dir):
    image_extensions = ('.png', '.jpg', '.jpeg')

    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue  # Skip files in the root folder

        image_count = 0
        non_image_count = 0

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if not filename.lower().endswith(image_extensions):
                os.remove(file_path)
                non_image_count += 1
            else:
                image_count += 1

        print(
            f"Folder '{folder_name}': Image files = {image_count}, "
            f"Non-image files removed = {non_image_count}"
        )
