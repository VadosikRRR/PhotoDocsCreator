import logging

from PIL import Image, ImageDraw, ImageFont
from pillow_heif import register_heif_opener

from models import Cropper, Model, MyBiRefNet

logger = logging.getLogger(__name__)


class PhotoDocsCreator:
    _BIREFNET_NAME = "BiRefNet"

    def __init__(self, seg_model_name: str = "BiRefNet", is_concat: bool = False):
        """This method creates an instance of the PhotoDocsCreator class

        Args:
            seg_model_name (str): The name of the model to use.
            is_concat (bool): Predicate whether to concatenate the source image and the result.
        """

        self.__del_background_model = self.__load_segmentation_model(seg_model_name)
        self.__cropper_model = self.__load_crop_image_model()
        self.__is_concat = is_concat
        register_heif_opener()

        logging.basicConfig(filename="myapp.log", level=logging.INFO)
        logger.info("\n✅ PhotoDocsCreator is ready")

    def __load_segmentation_model(self, seg_model_name: str) -> Model:
        """The method loads the segmentation model

        Args:
            seg_model_name (str): The name of the model to use.

        Returns:
            (Model): Returns the segmentation model
        """

        if seg_model_name == self._BIREFNET_NAME:
            return MyBiRefNet()

        raise NameError("The model with that name was not found")

    def __load_crop_image_model(self) -> Model:
        """The method loads the model for cropping photos.

        Returns:
            (Model): Returns the model for cropping photos.
        """

        return Cropper()

    def process(self, input_image: Image.Image) -> Image.Image:
        """The method processes the photo.

        Args:
            input_image (Image.Image): The input image.

        Returns:
            (Image.Image): Returns the finished photo.
        """

        logger.info("⏳ Start process. Please wait")
        croped_image = self.__cropper_model.predict(input_image)
        result = self.__del_background_model.predict(croped_image)
        if self.__is_concat:
            result = self.__concat_image(input_image, result)

        logger.info("✅ Complete!")
        return result

    def __concat_image(
        self, input_image: Image.Image, output_image: Image.Image
    ) -> Image.Image:
        """The method concatenates the source image and the result.

        Args:
            input_image (Image.Image): input image.
            output_image (Image.Image): result image.

        Returns:
            (Image.Image): concatenated image.
        """

        new_height = output_image.height
        new_width = int(input_image.width * new_height / input_image.height)
        input_image = input_image.resize(
            (new_width, new_height), Image.Resampling.LANCZOS
        )
        self.__add_text(input_image, "Input image")
        self.__add_text(output_image, "Output image")

        result_image = Image.new("RGB", (new_width + output_image.width, new_height))
        result_image.paste(input_image, (0, 0))
        result_image.paste(output_image, (new_width, 0))
        return result_image

    def __add_text(self, image: Image.Image, text: str):
        """The method adds text to a photo.

        Args:
            image (PIL.Image.Image): Input image.
            text (str): input text.
        """

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(image.height // 20)

        text_color = (255, 0, 0)
        position = (image.height // 20, image.height // 20)
        draw.text(position, text, fill=text_color, font=font)
