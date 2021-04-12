from fastapi import FastAPI
from pydantic import BaseModel
import blosc
import base64
import json

def encode_array(array, compressor=partial(blosc.pack_array, cname="lz4")):
    """
    Compressor numpy array to json-safe string
    """
    cdata = base64.urlsafe_b64encode(compressor(array)).decode("utf-8")
    return cdata


def decode_array(cdata, compressor=blosc.unpack_array):
    """
    Decode string to bytes and uncompress to numpy array
    """
    data = compressor(base64.urlsafe_b64decode(cdata))
    return data


app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = None


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
