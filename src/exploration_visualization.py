import os


def check_structure(data_dir, sets, labels):
    """
    Check integrity of the directory structure for the datasets.
    """
    for set_name in sets:
        for label in labels:
            path = os.path.join(data_dir, set_name, label)

            if not os.path.exists(path):
                print(f"Warning: '{label}' in '{set_name}' is missing.")
            elif not os.listdir(path):
                print(
                    f"Notice: '{label}' in '{set_name}' exists but is empty."
                    )
            else:
                image_count = len(os.listdir(path))
                print(
                    f"'{label}' in '{set_name}' is valid with "
                    f"{image_count} images."
                )
