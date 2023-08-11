from huggingface_hub import HfApi
from huggingface_hub import snapshot_download

api = HfApi()

snapshot_download(repo_id="chao1224/Geom3D_data", repo_type="dataset", local_dir=".")
