import os
import warnings

from PIL import Image

from PhotoDocsCreator import PhotoDocsCreator

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    pdc = PhotoDocsCreator(is_concat=True)
    project_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    image = Image.open(os.path.join(project_dir, "src/tests", "Example.jpeg"))
    pdc.process(image).save(os.path.join(project_dir, "src/tests", "ResultExample.jpg"))
