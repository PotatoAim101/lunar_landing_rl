from pathlib import Path

agents_names = ["ddpg/", "pg/", "actor-critic/"]

# Folders and file paths
project_path = Path(__file__).parent.resolve()

model_folder = project_path / "models/"
model_folder.mkdir(parents=True, exist_ok=True)
for name in agents_names:
    tmp = model_folder / name
    tmp.mkdir(parents=True, exist_ok=True)

video_folder = project_path / "tmp/video/"
video_folder.mkdir(parents=True, exist_ok=True)
for name in agents_names:
    tmp = video_folder / name
    tmp.mkdir(parents=True, exist_ok=True)

plots_folder = project_path / 'plots/'
plots_folder.mkdir(parents=True, exist_ok=True)
for name in agents_names:
    tmp = plots_folder / name
    tmp.mkdir(parents=True, exist_ok=True)