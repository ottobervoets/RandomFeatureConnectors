from models.base_rfc import BaseRFC
from models.PCA_rfc import PCARFC

def create_RFC(type = "base", **kwargs):
    match(type):
        case "base":
            return BaseRFC(**kwargs)
        case "PCARFC":
            return PCARFC(**kwargs)
        case _:
            Exception("type not availible")