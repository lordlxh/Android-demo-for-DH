import os

folder_path = "full_body_img"

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, filename.replace(".jpg", ".sij"))
        os.rename(old_file_path, new_file_path)
        print(f"Renamed: {old_file_path} -> {new_file_path}")

print("All .jpg to .sij.")
