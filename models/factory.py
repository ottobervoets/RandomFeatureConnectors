from models.PCA_rfc import PCARFC
from models.randomG_rfc import RandomGRFC
from models.base_rfc import BaseRFC
from models.base_adaptive import Base_adaptive
from models.Base_2 import Base_2
from models.reservoir_sample_rfc import ReservoirSampleRFC
# from models.matrix_conceptor import MatrixConceptor
from models.matrix_conceptor_rebuild import MatrixConceptor
from matlab_copy.matrix_conceptor_matlab import MatrixConceptorMatlab
from matlab_copy.general_setup_class import MatriXConceptorWorking

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
        case "reservoir_sample_rfc":
            return ReservoirSampleRFC(**kwargs)
        case "matrix_conceptor":
            return MatrixConceptor(**kwargs)
            # return MatrixConceptorMatlab(**kwargs)
            # return MatriXConceptorWorking(**kwargs)
        case _:
            Exception("type not available")
