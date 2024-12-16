# from ts.torch_handler.base_handler import BaseHandler
import os
import zipfile
import logging
from ts.torch_handler.vision_handler import VisionHandler

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class LatexHandler(VisionHandler):

    def initialize(self, context):
        print(f"Initializing model from {context=}")
        logger.info(f"Initializing model from {context=}")
        super().initialize(context)
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # REF: https://github.com/pytorch/serve/issues/1227 
        with zipfile.ZipFile(model_dir + '/nougat.zip', 'r') as zip_ref:
            zip_ref.extractall(model_dir)

        from nougat_latex import NougatLaTexProcessor

        model_name = "Norm/nougat-latex-base"
        self.latex_processor = NougatLaTexProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.decoder_input_ids = self.tokenizer(
            self.tokenizer.bos_token, add_special_tokens=False, return_tensors="pt"
        ).input_ids

    def image_processing(self, image):
        return self.latex_processor(image, return_tensors="pt")['pixel_values']

    def preprocess(self, data):
        print(f"Preprocess {data=}")
        # image = Image.open("sample-images/lt-2.jpg")
        # if not image.mode == "RGB":
        #     image = image.convert('RGB')

        return super().preprocess()     # data)

    def postprocess(self, data):
        print(f"Postprocess {data=}")
        return super().postprocess(data)
