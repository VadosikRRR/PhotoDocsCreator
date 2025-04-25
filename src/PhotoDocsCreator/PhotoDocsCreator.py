from PIL import Image, ImageDraw, ImageFont
from pillow_heif import register_heif_opener

from models import BiRefNet, Croper, Model


class PhotoDocsCreator:
    _BIREFNET_NAME = "BiRefNet"

    def __init__(self, seg_model_name: str = "BiRefNet", is_concat: bool = False):
        """This method creates an instance of the PhotoDocsCreator class

        Args:
            seg_model_name (str): The name of the model to use.
            is_concat (bool): Predicate whether to concatenate the source image and the result.
            offset_from_top (float): Offset relative to the top in a fraction
        """

        self.__del_background_model = self.__load_segmentation_model(seg_model_name)
        self.__croper_model = self.__load_crop_image_model()
        self.__is_concat = is_concat
        register_heif_opener()
        print("\n✅ PhotoDocsCreator is ready")

    def __load_segmentation_model(self, seg_model_name: str) -> Model:
        """The method loads the segmentation model

        Args:
            seg_model_name (str): The name of the model to use.

        Returns:
            (Model): Returns the segmentation model
        """

        if seg_model_name == self._BIREFNET_NAME:
            return BiRefNet()

        return None

    def __load_crop_image_model(self) -> Model:
        """The method loads the model for cropping photos.

        Returns:
            (Model): Returns the model for cropping photos.
        """

        return Croper()

    def process(self, path_to_image: str, path_to_save: str, name_image: str):
        """The method processes the photo.

        Args:
            path_to_image (str): The path to photo.
            path_to_save (str): The path to the place of preservation.
            name_image (str): Name of the photo to save.
        """

        print("⏳ Start process. Please wait")

        input_image = Image.open(path_to_image)
        croped_image = self.__croper_model.predict(input_image)
        result = self.__del_background_model.predict(croped_image)
        if self.__is_concat:
            result = self.__concat_image(input_image, result)

        result.save(path_to_save + "/PhotoDocsCreator" + name_image + ".jpg")

        print("✅ Complete!")

    def __concat_image(self, input_image: Image.Image, output_image: Image.Image):
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

    def __add_text(self, image, text: str):
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
