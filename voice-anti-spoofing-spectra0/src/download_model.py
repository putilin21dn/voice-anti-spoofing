from pathlib import Path
from huggingface_hub import hf_hub_download

root = Path(__file__).resolve().parents[1]
dest = root / "models" / "spectra_0"
dest.mkdir(parents=True, exist_ok=True)

for filename in ("model.py", "model.safetensors"):
    path = hf_hub_download(
        repo_id="lab260/spectra_0",
        filename=filename,
        local_dir=str(dest),
    )
    print("Downloaded:", path)

print("The XLS-R encoder is downloaded automatically on first inference.")
