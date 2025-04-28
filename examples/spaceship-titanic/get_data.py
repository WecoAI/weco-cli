import kaggle
import zipfile
import os


def get_data():
    kaggle.api.competition_download_files("spaceship-titanic")
    # unzip the data
    with zipfile.ZipFile("spaceship-titanic.zip", "r") as zip_ref:
        zip_ref.extractall("data")
    # remove the zip file
    os.remove("spaceship-titanic.zip")


if __name__ == "__main__":
    get_data()
