# phishserve/handler.py
import os, json, torch, torch.nn.functional as F
from ts.torch_handler.base_handler import BaseHandler
from model import PhishingClassifier
import re

def clean_text(text):
    # remove protocol
    text = re.sub(r'^https?://', '', text)
    # remove www
    text = re.sub(r'^www.', '', text)
    # split by special characters
    tokens = re.split(r'[/.?=_-]', text)
    # remove empty tokens
    tokens = [token for token in tokens if token]
    return tokens

class PhishHandler(BaseHandler):
    def initialize(self, ctx):
        self.manifest = ctx.manifest
        model_dir = ctx.system_properties.get("model_dir")
        self.device = torch.device("cuda" if torch.cuda.is_available() and ctx.system_properties.get("gpu_id") is not None else "cpu")

        # load vocab
        with open(os.path.join(model_dir, "itos.txt")) as f:
            self.itos = [line.strip() for line in f]
        self.stoi = {s:i for i,s in enumerate(self.itos)}
        
        self.PAD = self.stoi["<pad>"]
        self.UNK = self.stoi["<unk>"]

        # load model
        self.model = self.load_model(model_dir)
        self.model.eval()
        
        # load max_len from checkpoint
        ck = torch.load(os.path.join(model_dir, "best.pt"), map_location=self.device)
        self.max_len = ck["max_len"]


    def load_model(self, model_dir):
        # load state dict from the serialized model file
        state_dict = torch.load(os.path.join(model_dir, "model.pt"), map_location=self.device)
        
        # load checkpoint to get model args
        ck = torch.load(os.path.join(model_dir, "best.pt"), map_location=self.device)
        
        model = PhishingClassifier(vocab_size=len(self.itos), emb_dim=ck["emb_dim"], hid=ck["hid"], num_classes=2, pad_idx=self.PAD)
        model.load_state_dict(state_dict)
        model.to(self.device)
        return model

    def _encode(self, text: str):
        toks = clean_text(text)
        ids = [self.stoi.get(t, self.UNK) for t in toks][:self.max_len]
        if len(ids) < self.max_len:
            ids += [self.PAD] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def preprocess(self, data):
        texts = []
        for item in data:
            body = item.get("body", item.get("data", item))
            if isinstance(body, (bytes, bytearray)): body = body.decode("utf-8")
            
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except json.JSONDecodeError:
                    texts.append(body)
                    continue

            if isinstance(body, dict):
                texts.append(body.get("url"))

        batch = torch.stack([self._encode(t) for t in texts], dim=0).to(self.device, non_blocking=True)
        return batch

    def inference(self, model_input):
        with torch.inference_mode():
            logits = self.model(model_input)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
        return pred, conf

    def postprocess(self, inference_output):
        pred, conf = inference_output
        labels = ["benign", "phish"]
        return [{"label": labels[p], "confidence": float(c)} for p, c in zip(pred, conf)]
