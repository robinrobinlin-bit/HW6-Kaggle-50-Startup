import os
import urllib.request
import zipfile
import shutil

def download_and_extract(url, zip_name, extract_to):
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, zip_name)
    print(f"Extracting {zip_name} to {extract_to}...")
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_name)
    print("Done!")

if __name__ == "__main__":
    workspace = os.path.dirname(os.path.abspath(__file__))
    env_dir = os.path.join(workspace, "env")
    
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
        
    # Download Node.js v22.14.0 (satisfies 22+ requirement)
    node_url = "https://nodejs.org/dist/v22.14.0/node-v22.14.0-win-x64.zip"
    node_zip = os.path.join(workspace, "node.zip")
    node_extract = os.path.join(env_dir, "node")
    if not os.path.exists(node_extract):
        download_and_extract(node_url, node_zip, node_extract)
        
    # Download FFmpeg
    ffmpeg_url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    ffmpeg_zip = os.path.join(workspace, "ffmpeg.zip")
    ffmpeg_extract = os.path.join(env_dir, "ffmpeg")
    if not os.path.exists(ffmpeg_extract):
        download_and_extract(ffmpeg_url, ffmpeg_zip, ffmpeg_extract)
        
    print("Environment setup script completed.")
