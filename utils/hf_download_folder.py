import os
from huggingface_hub import hf_hub_download, HfApi
import os
import time
import concurrent.futures


def _download_file_with_retry(args):
    hf_token, repo_id, repo_type, filename, local_folder_path, remote_folder_path_for_logging = args

    while True:
        try:
            local_file_path = hf_hub_download(
                token=hf_token,
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
                local_dir=local_folder_path
            )
            return local_file_path
        except Exception as e:
            print(f"Error downloading {repo_id}: {remote_folder_path_for_logging}/{filename}: {e}")
            time.sleep(1)


def download_folder(hf_token, repo_id, repo_type, remote_folder_path, local_folder_path, num_processes):
    os.makedirs(local_folder_path, exist_ok=True)
    
    api = HfApi(token=hf_token)
    # List all files in the repository
    all_repo_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)

    # Ensure subfolder path has no leading slash
    if remote_folder_path.startswith("/"):
        remote_folder_path = remote_folder_path[1:]

    files_to_download = [f for f in all_repo_files if f.startswith(remote_folder_path)]

    if not files_to_download:
        print(f"No files found in {repo_id}: {remote_folder_path}")
        return

    # Adjust num_processes if it's greater than the number of files
    actual_num_processes = min(num_processes, len(files_to_download))
    if actual_num_processes <= 0: # handle case with no files or invalid num_processes
        actual_num_processes = 1


    downloaded_files = []
    
    # Prepare arguments for each download task
    tasks_args = [
        (hf_token, repo_id, repo_type, f, local_folder_path, remote_folder_path) for f in files_to_download
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=actual_num_processes) as executor:
        results = list(executor.map(_download_file_with_retry, tasks_args))
        downloaded_files = [result for result in results if result]


    print(f"Successfully downloaded {len(downloaded_files)} files from {repo_id}: {remote_folder_path} using {actual_num_processes} processes. Downloaded files:")
    for file_path in downloaded_files:
        print(file_path)



if __name__ == "__main__":
    # Example usage
    HF_TOKEN = os.environ["HF_TOKEN"]
    CACHE_ROOT = f"The cache root"
    REPO_ID = "xxx/xxx"
    repo_type = "model"
    remote_folder_paths = [
        f"xxx/xxx",
    ]
    local_folder_path = f"{CACHE_ROOT}/xxx"
    num_download_processes = 64 # User-defined number of processes
    for remote_folder_path in remote_folder_paths:
        download_folder(HF_TOKEN, REPO_ID, repo_type, remote_folder_path, local_folder_path, num_download_processes)
    