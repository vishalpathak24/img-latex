# from ts.torch_handler.base_handler import BaseHandler
import os
import io
import zipfile
import logging
import torch
import numpy as np
#from ts.torch_handler.vision_handler import VisionHandler
from ts.torch_handler.base_handler import BaseHandler
from transformers import VisionEncoderDecoderModel
from transformers import AutoTokenizer
from PIL import Image

logger = logging.getLogger(__name__)


class LatexHandler(BaseHandler):
    
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize(self, context):
        print(f"Initializing model from {context.__dict__=}")
        logger.info(f"Initializing model from {context.__dict__=}")
        self.context = context
        model_dir = context.system_properties.get("model_dir")
        self.model = VisionEncoderDecoderModel.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        
        #super().initialize(context)
        # properties = context.system_properties
        # model_dir = properties.get("model_dir")

        # REF: https://github.com/pytorch/serve/issues/1227 
        
        with zipfile.ZipFile(model_dir + '/nougat.zip', 'r') as zip_ref:
            zip_ref.extractall(model_dir+'/nougat_latex')

        from nougat_latex import NougatLaTexProcessor

        model_name = "Norm/nougat-latex-base"
        self.latex_processor = NougatLaTexProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.decoder_input_ids = self.tokenizer(
            self.tokenizer.bos_token, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)
    

    def preprocess(self, data):
        print(f"Preprocess {data=}")
        img_data = []
        for d in data:
            image = Image.open(io.BytesIO(d['data']))
            px_datas = self.latex_processor(image).pixel_values
            img_data.append(px_datas[0])
        # image = Image.open("sample-images/lt-2.jpg")
        # if not image.mode == "RGB":
        #     image = image.convert('RGB')
        
        return super().preprocess(np.array(img_data))
    
    def inference(self, data, *args, **kwargs):
        batch_sz = data.shape[0]
        print("Batch Size", batch_sz)
        decoder_strt_inputs = torch.tensor(
            np.array([self.decoder_input_ids[0]]*batch_sz)
        )
        
        with torch.inference_mode():
            marshalled_data = data.to(self.device)
            results = self.model.generate(
                marshalled_data,
                decoder_input_ids=decoder_strt_inputs,
                max_length=self.model.decoder.config.max_length,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=5,
                bad_words_ids=[[self.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )
        return results

    def postprocess(self, data):
        #print(f"Postprocess {data=}")
        sequence = self.tokenizer.batch_decode(data.sequences)
        sequence = [ 
            s.replace(self.tokenizer.eos_token, "").replace(self.tokenizer.pad_token, "").replace(self.tokenizer.bos_token, "")
            for s in sequence
        ]
        print(sequence)
        return sequence #super().postprocess(sequence)
