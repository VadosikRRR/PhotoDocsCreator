import os
from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from BiRefNet.models.birefnet import BiRefNet
from face_parsing.model import BiSeNet


class CoordinatesError(Exception):
    def __init__(self, message, value):
        self.message = message
        super().__init__(self.message)


class Model(ABC):
    def __init__(self):
        self.__model = None
        self._prepare_model()

    @abstractmethod
    def _prepare_model(self):
        """
        This method prepares the model.
        """

        pass

    @abstractmethod
    def predict(self, image: Image.Image) -> Image.Image:
        """This method applies the model to the image.

        Args:
            image (Image.Image): input image.

        Returns:
            (Image.Image): processed image.
        """

        pass


class MyBiRefNet(Model):
    def _prepare_model(self):
        torch.set_float32_matmul_precision("high")
        self.__model = BiRefNet.from_pretrained("ZhengPeng7/BiRefNet")
        self.__model.to("cpu")
        self.__model.eval()
        self.__image_size = (1024, 1024)
        self.__transform_image = v2.Compose(
            [
                v2.Resize(self.__image_size),
                v2.ToTensor(),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __segment_image(self, image: Image.Image) -> Image.Image:
        """This method segments the image.

        Args:
            image (Image.Image): input image.

        Returns:
            (Image.Image): returns the segmentation result
        """

        input_images = self.__transform_image(image).unsqueeze(0)
        with torch.no_grad():
            preds = self.__model(input_images)[-1].sigmoid().cpu()

        pred = preds[0].squeeze()
        pred_pil = v2.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        image.putalpha(mask)
        return image

    def predict(self, image):
        """The method segments the image and creates a white background.

        Args:
            image (Image.Image): input image.

        Returns:
            (Image.Image): replaces the background with white.
        """

        segmented_image = self.__segment_image(image)
        out_image = np.ones((image.height, image.width, 3), dtype=np.uint8) * 255
        out_image = Image.fromarray(out_image)
        out_image.paste(segmented_image, mask=segmented_image.split()[3])
        return out_image


class Cropper(Model):
    def _prepare_model(self):
        self.__n_classes = 19
        self.__model = BiSeNet(n_classes=self.__n_classes)
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        save_pth = os.path.join(script_dir + "/face_parsing/res/cp", "79999_iter.pth")
        self.__model.load_state_dict(
            torch.load(save_pth, map_location=torch.device("cpu"))
        )
        self.__model.eval()
        self.__transform_image = v2.Compose(
            [
                v2.ToTensor(),
                v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.__upper_face_height_factor = 0.2007
        self.__under_face_height_factor = -0.1914
        self.__upper_hair_height_factor = 0.2090
        self.__under_neck_height_factor = 0.5624
        self.__width_to_height_ratio = 35 / 45

    def __make_image_to_squre(self, image: Image.Image) -> Image.Image:
        """The method crops the photo to fit a square one.

        Args:
            image (Image.Image): the original photo, Image.Image (height, width)

        Returns:
            (Image.Image): a square photo, Image.Image (min(height, width), min(height, width))
        """

        height = image.height
        width = image.width
        mask = self.__get_mask_from_image(image)

        mask_resized = cv2.resize(
            mask, (width, height), interpolation=cv2.INTER_NEAREST
        )

        # 15 - 18 --- not important (number : body part)
        mask_resized[mask_resized >= 15] = 0
        face_coordinates = np.argwhere(mask_resized >= 1)
        if face_coordinates.size == 0:
            return self.__easy_make_to_square(image)

        start_excess = 0
        end_excess = 0
        if height >= width:
            real_height = face_coordinates[:, 0]
            y_center = int(np.median(real_height))
            y_start = y_center - width // 2
            if y_start < 0:
                start_excess = abs(y_start)
                y_start = 0

            y_end = y_center + width // 2
            if y_end > height:
                end_excess = y_end - height
                y_end = height

            return image.crop((0, y_start - end_excess, width, y_end + start_excess))

        real_width = face_coordinates[:, 1]
        x_center = int(np.median(real_width))
        x_start = x_center - height // 2
        if x_start < 0:
            start_excess = abs(x_start)
            x_start = 0

        x_end = x_center + height // 2
        if x_end > width:
            end_excess = x_end - width
            x_end = width

        return image.crop((x_start - end_excess, 0, x_end + start_excess, height))

    def __easy_make_to_square(self, image: Image.Image) -> Image.Image:
        """The method crops the photo to fit a square one.

        Args:
            image (Image.Image): the original photo, Image.Image (height, width)

        Returns:
            (Image.Image): a square photo, Image.Image (min(height, width), min(height, width))
        """
        height = image.height
        width = image.width
        min_length = min(height, width)
        dh = (height - min_length) // 2
        dw = (width - min_length) // 2
        return image.crop((dw, dh, dw + min_length, dh + min_length))

    def __get_mask_from_image(self, image: Image.Image) -> np.array:
        """The method returns a mask with different parts of the face.

        Args:
            image (Image.Image): square image.

        Returns:
            (np.array): to return a mask with different parts of the face.
        """

        image = image.resize((512, 512), Image.BILINEAR)
        image = self.__transform_image(image)
        image = torch.unsqueeze(image, 0)

        with torch.no_grad():
            out = self.__model(image)[0]
            mask = out.squeeze(0).cpu().numpy().argmax(0)

        return mask

    def __get_coordintates_from_face(self, mask: np.array) -> tuple[int, int, int, int]:
        """The method finds the coordinates for cropping a photo using a mask (Face search).

        Args:
            mask (np.array): The mask of the input image. The size of the mask must match the size of the image.

        Returns:
            (tuple[int, int, int, int]): Return coordinates received from a face.
        """

        # 1 - face
        if 1 not in mask:
            raise CoordinatesError("The face could not be recognized")

        size_image = mask.shape[0]
        face_coordinates = np.argwhere(mask == 1)

        start_face_index = face_coordinates[0, 0]
        end_face_index = face_coordinates[-1, 0]
        face_height = end_face_index - start_face_index
        start_image_height_index = int(
            start_face_index - face_height * self.__upper_face_height_factor
        )
        end_image_height_index = int(
            end_face_index + face_height * self.__under_face_height_factor
        )
        end_image_height_index = (
            end_image_height_index
            if end_image_height_index <= size_image
            else size_image
        )

        face_width_index = face_coordinates[:, 1]
        center_of_face_width = np.median(face_width_index)
        width_image = int(
            (end_image_height_index - start_image_height_index)
            * self.__width_to_height_ratio
        )
        start_image_width_index = int(center_of_face_width - width_image * 0.5)
        start_image_width_index = (
            start_image_width_index if start_image_width_index >= 0 else 0
        )
        end_image_width_index = int(center_of_face_width + width_image * 0.5)
        end_image_width_index = (
            end_image_width_index if end_image_width_index <= size_image else size_image
        )

        return (
            start_image_width_index,
            start_image_height_index,
            end_image_width_index,
            end_image_height_index,
        )

    def __get_coordinates_from_hair_and_neck(
        self, mask: np.array
    ) -> tuple[int, int, int, int]:
        """The method finds the coordinates for cropping a photo using a mask (Neck and hair search)

        Args:
            mask (np.array): The mask of the input image. The size of the mask must match the size of the image.

        Returns:
            (tuple[int, int, int, int]): Return coordinates received from a hair and neck.
        """

        # 14 - neck, 17 - hair
        if 14 not in mask or 17 not in mask:
            raise CoordinatesError("Neck or hair could not be recognized")

        hair_coordinates = np.argwhere(mask == 17)
        neck_coordinates = np.argwhere(mask == 14)

        end_image_height_index = neck_coordinates[-1, 0]
        start_image_height_index = hair_coordinates[0, 0]

        image_height = end_image_height_index - start_image_height_index + 1
        start_image_height_index = int(
            start_image_height_index - image_height * self.__upper_hair_height_factor
        )

        end_image_height_index = int(
            end_image_height_index + image_height * self.__under_neck_height_factor
        )

        image_height = end_image_height_index - start_image_height_index + 1

        hair_width_indexes = hair_coordinates[:, 1]
        neck_width_indexes = neck_coordinates[:, 1]
        width_center_of_hair = np.median(hair_width_indexes)
        width_center_of_neck = np.median(neck_width_indexes)
        width_image_center = int((width_center_of_hair + width_center_of_neck) / 2)

        image_width = int(image_height * self.__width_to_height_ratio)
        start_image_width_index = int(width_image_center - image_width / 2)
        end_image_width_index = int(width_image_center + image_width / 2)

        return (
            start_image_width_index,
            start_image_height_index,
            end_image_width_index,
            end_image_height_index,
        )

    def __results_coordinates(
        self, coordinates: list[tuple[int, int, int, int]], width_image: int
    ):
        """The method calculates the average for all coordinates.

        Args:
            coordinates (list[tuple[int, int, int, int]]): An array with the final coordinates.
            width_image (int): image width

        Returns:
            (tuple[int, int, int, int]): Return the final coordinates for clipping.
        """

        mean_y_start = 0
        mean_y_end = 0
        mean_x_center = 0
        for x1, y1, x2, y2 in coordinates:
            mean_y_start += y1
            mean_y_end += y2
            mean_x_center += (x1 + x2) / 2

        mean_y_start = int(mean_y_start / len(coordinates))
        mean_y_end = int(mean_y_end / len(coordinates))
        height = mean_y_end - mean_y_start + 1
        width = int(height * self.__width_to_height_ratio)
        mean_x_center = int(mean_x_center / len(coordinates))
        mean_x_start = int(mean_x_center - width / 2)
        mean_x_end = int(mean_x_center + width / 2)
        mean_x_start = mean_x_start if mean_x_start >= 0 else 0
        mean_x_end = mean_x_end if mean_x_end <= width_image else width_image

        return mean_x_start, mean_y_start, mean_x_end, mean_y_end

    def predict(self, image):
        """The method crops the photo and adds white borders if there is not enough image.

        Args:
            image (Image.Image): square image.

        Return:
            (Image.Image): cropped image.
        """

        image = self.__make_image_to_squre(image)
        coordinates = []
        mask = self.__get_mask_from_image(image)
        mask_resized = cv2.resize(
            mask, (image.width, image.height), interpolation=cv2.INTER_NEAREST
        )

        coordinates.append(self.__get_coordintates_from_face(mask_resized))
        coordinates.append(self.__get_coordinates_from_hair_and_neck(mask_resized))

        (x1, y1, x2, y2) = self.__results_coordinates(coordinates, image.height)

        if y1 >= 0:
            return image.crop((x1, y1, x2, y2))

        croped_image = image.crop((x1, 0, x2, y2))
        additive = Image.new(
            mode="RGB", size=(croped_image.width, abs(y1)), color=(255, 255, 255)
        )
        result = Image.new(
            mode="RGB", size=(croped_image.width, additive.height + croped_image.height)
        )
        result.paste(additive, (0, 0))
        result.paste(croped_image, (0, additive.height))
        return result
