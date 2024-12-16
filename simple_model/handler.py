import json
from ts.torch_handler.base_handler import BaseHandler


class MyHandler(BaseHandler):
    image_processing = [
    ]
    
    def preprocess(self, data):
        print(f"Preprocess Got data as {data=}")
        body_data = json.loads(data[0]['body'])
        return super().preprocess(body_data)
    
    def postprocess(self, data):
        print(f"Post Processing {data=}")
        return super().postprocess(data)
