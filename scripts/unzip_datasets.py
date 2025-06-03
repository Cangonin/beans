import zipfile
from pathlib import Path
from typing import List, Optional


def get_all_zip_paths(root_folder: Path) -> List[Path]:
    zip_file_paths = list(root_folder.glob("*.zip"))
    assert len(zip_file_paths) == 12
    return zip_file_paths


def unzip_dataset(dataset_zip_path: Path) -> None:
    with zipfile.ZipFile(dataset_zip_path, "r") as zip_ref:
        print(f"Extracting {dataset_zip_path.name}...")
        zip_ref.extractall(path=dataset_zip_path.parent)
    print(f"{dataset_zip_path} successfully extracted")


def unzip_all_datasets(root_folder: Path) -> None:
    all_zip_paths = get_all_zip_paths(root_folder)
    for zip_path in all_zip_paths:
        unzip_dataset(zip_path)


if __name__ == "__main__":
    root_folder = Path(__file__).resolve().parent.parent / "data"
    unzip_all_datasets(root_folder)
