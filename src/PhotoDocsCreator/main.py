import os
import warnings

from PhotoDocsCreator import PhotoDocsCreator

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    pdc = PhotoDocsCreator(is_concat=True)
    project_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    test_dir = project_dir + "/src/tests"
    pdc.process(test_dir + "/Example.jpeg", test_dir, "ResultExample")
