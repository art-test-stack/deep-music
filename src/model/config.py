class TextEncoderConfig:
    def __init__(self, model_name: str, model_type: str, model_path: str):
        self.model_name = model_name
        self.model_type = model_type
        self.model_path = model_path

class AudioEncoderConfig:
    def __init__(self, model_name: str, model_type: str, model_path: str):
        self.model_name = model_name
        self.model_type = model_type
        self.model_path = model_path

class DecoderConfig:
    def __init__(self, model_name: str, model_type: str, model_path: str):
        self.model_name = model_name
        self.model_type = model_type
        self.model_path = model_path
        