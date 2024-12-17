from models.PCA_rfc import PCARFC
from models.randomG_rfc import RandomGRFC
from models.base_rfc import BaseRFC
from models.base_adaptive import Base_adaptive
from models.Base_2 import Base_2

def create_RFC(rfc_type = "base", **kwargs):
    match(rfc_type):
        case "base":
            return BaseRFC(**kwargs)
        case "PCARFC":
            return PCARFC(**kwargs)
        case "randomG":
            return RandomGRFC(**kwargs)
        case "base_adaptive":
            return Base_adaptive(**kwargs)
        case "base_2":
            return Base_2(**kwargs)
        case _:
            Exception("type not available")
