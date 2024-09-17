import os
import glob


def keep_only_ten_images(data_dir):
    # Iterate through each subdirectory in the 'data' directory
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)

        # Check if it's a directory
        if os.path.isdir(subdir_path):
            # Get all PNG images in the subdirectory
            image_files = glob.glob(os.path.join(subdir_path, "*.png"))

            # Sort the image files for consistency (e.g., by name)
            image_files.sort()

            # If more than 10 images exist, remove the excess images
            if len(image_files) > 10:
                for image_path in image_files[10:]:
                    try:
                        os.remove(image_path)  # Remove the image file
                        print(f"Removed {image_path}")
                    except Exception as e:
                        print(f"Error removing {image_path}: {e}")


# Example usage
data_directory = "./categorized_spectrograms"  # Replace with the path to your data directory
keep_only_ten_images(data_directory)
