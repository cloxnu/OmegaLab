import pathlib
import os

dataset_dir = os.path.dirname(__file__)

supported_image_type = ['*.jpg', '*.jpeg', '*.png']


def load_image_path(names: [str]) -> dict:
    image_path_label = {}
    for idx, name in enumerate(names):
        data_path = pathlib.Path(dataset_dir)
        data_path = data_path / name
        if not data_path.exists():
            raise Exception('NameError: There is no data named "{}"'.format(name))

        for t in supported_image_type:
            image_path_label.update({str(path): idx for path in data_path.glob(t)})

    return image_path_label
