from models.base_rfc import BaseRFC
from models.PCA_rfc import PCARFC
from models.randomG_rfc import RandomGRFC

def create_RFC(rfc_type = "base", **kwargs):
    match(rfc_type):
        case "base":
            return BaseRFC(**kwargs)
        case "PCARFC":
            return PCARFC(**kwargs)
        case "randomG":
            return RandomGRFC(**kwargs)
        case _:
            Exception("type not available")
