from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE, REPO_TYPES
from huggingface_hub.file_download import repo_folder_name
import os
import shutil
import zipnn
import json

from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)

def remove_znn_extension(file_path):
    # Step 1: Open and read the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Step 2: Modify only the `weight_map` section to remove the ".znn" extension
    for key, value in data.get("weight_map", {}).items():
        if value.endswith('.znn'):
            # Replace the '.znn' extension in the file name
            data['weight_map'][key] = value.replace('.znn', '')

    # Step 3: Save the modified data back to the file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Updated {file_path} by removing '.znn' extensions.")

def update_link(directory,znn_file):
    znn_path = os.path.join(directory, znn_file)

    # Get the target of the symlink
    target_file = os.readlink(znn_path)

    # Find the corresponding .safetensors file
    safetensors_file = znn_file.replace('.znn', '')
    safetensors_path = os.path.join(directory, safetensors_file)

    # If the .safetensors file exists, replace the target file with it
    if os.path.exists(safetensors_path):
        target_full_path = os.path.join(directory, target_file)

        # Move the .safetensors file to the actual target file
        shutil.move(safetensors_path, target_full_path)

        # Recreate the symlink to the updated target file
        os.symlink(target_file, safetensors_path)

        # Remove the old symlink
        os.remove(znn_path)
        print(f"Updated link for {znn_file}: {safetensors_file} -> {target_file}")
    else:
        print(f".safetensors file not found for {znn_file}")



def decompress_file(input_file, dtype="", delete=False, force=False):

    if not input_file.endswith(".znn"):
        raise ValueError("Input file does not have the '.znn' suffix")

    if os.path.exists(input_file):
        if delete:
            print(f"Deleting {input_file}...")
            os.remove(input_file)
        else:
            decompressed_path = input_file[:-4]
            if not force and os.path.exists(decompressed_path):

                user_input = (
                    input(f"{decompressed_path} already exists; overwrite (y/n)? ").strip().lower()
                )

                if user_input not in ("yes", "y"):
                    print(f"Skipping {input_file}...")
                    return
            print(f"Decompressing {input_file}...")

            output_file = input_file[:-4]

            if dtype:
                zpn = zipnn.ZipNN(is_streaming=True, bytearray_dtype="float32")
            else:
                zpn = zipnn.ZipNN(is_streaming=True)

            with open(input_file, "rb") as infile, open(output_file, "wb") as outfile:
                d_data = b""
                chunk = infile.read()
                d_data += zpn.decompress(chunk)
                outfile.write(d_data)
                print(f"Decompressed {input_file} to {output_file}")

    else:
        print(f"Error: The file {input_file} does not exist.")



def decompress_zpn_files(
    dtype="",
    path=".",
    delete=False,
    force=False,
    max_processes=1,
):

    file_list = []
    file_name_list = []
    root_list = []
    for root, dirs, files in os.walk(path):
        for file_name in files:
            if file_name == "model.safetensors.index.json":
                remove_znn_extension(os.path.join(root, file_name))
            if file_name.endswith(".znn"):
                # Find the corresponding .safetensors file
                safetensors_file = file_name.replace('.znn', '')
                safetensors_path = os.path.join(root, safetensors_file)

                # If the .safetensors file exists, replace the target file with it
                if os.path.exists(safetensors_path):
                    os.remove(os.path.join(root, file_name))
                    continue

                decompressed_path = file_name[:-4]
                if not force and os.path.exists(
                    decompressed_path
                ):
                    user_input = (
                        input(
                            f"{decompressed_path} already exists; overwrite (y/n)? "
                        )
                        .strip()
                        .lower()
                    )
                    if user_input not in (
                        "y",
                        "yes",
                    ):
                        print(
                            f"Skipping {file_name}..."
                        )
                        continue
                full_path = os.path.join(
                    root,
                    file_name,
                )
                file_list.append(full_path)
                file_name_list.append(file_name)
                root_list.append(root)



    with ProcessPoolExecutor(
        max_workers=max_processes
    ) as executor:

            future_to_file = {
                executor.submit(
                    decompress_file,
                    file,
                    dtype,
                    delete,
                    True,
                ): file
                for file in file_list[
                    :max_processes
                ]
            }

            file_list = file_list[max_processes:]
            while future_to_file:

                for future in as_completed(
                    future_to_file
                ):
                    file = future_to_file.pop(
                        future
                    )
                    try:
                        future.result()
                    except Exception as exc:
                        print(
                            f"File {file} generated an exception: {exc}"
                        )

                    if file_list:
                        next_file = file_list.pop(
                            0
                        )
                        future_to_file[
                            executor.submit(
                                decompress_file,
                                next_file,
                                dtype,
                                delete,
                                True,
                            )
                        ] = next_file
                        #


    for file_name, root in zip(file_name_list, root_list):
        update_link(root,file_name)

def decompress_model(repo_id):
    cache_dir=None
    repo_type= None
    # Replace "model_name" with the actual model name you want to download
    model_path = snapshot_download(repo_id=repo_id)
    if cache_dir is None:
        cache_dir = HF_HUB_CACHE

    if repo_type is None:
        repo_type = "model"
    if repo_type not in REPO_TYPES:
        raise ValueError(f"Invalid repo type: {repo_type}. Accepted repo types are: {str(REPO_TYPES)}")
    storage_folder = os.path.join(cache_dir, repo_folder_name(repo_id=repo_id, repo_type=repo_type))
    decompress_zpn_files(path=storage_folder)
    print(f"Model downloaded to: {model_path}")


if __name__ == "__main__":
    decompress_model(repo_id="royleibov/granite-7b-instruct-ZipNN-Compressed")