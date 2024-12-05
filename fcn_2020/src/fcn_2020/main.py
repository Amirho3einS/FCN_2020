from epilepsy2bids.annotations import Annotations
from epilepsy2bids.eeg import Eeg

from .utils import get_conversion_order, preprocess_ch
from .fcn import fcn_algorithm


def main(edf_file, outFile):

    eeg = Eeg.loadEdfAutoDetectMontage(edfFile=edf_file)
    if eeg.montage is Eeg.Montage.UNIPOLAR:
        eeg.reReferenceToBipolar()

    assert eeg.fs == 256, "Sampling frequency must be 256 Hz"

    # Get the conversion order
    conversion_order = get_conversion_order(eeg.channels)
    reoredered_data = eeg.data[conversion_order]
    data = preprocess_ch(reoredered_data, eeg.fs)
    # return data
    hypMask = fcn_algorithm(data, eeg.fs)
    hyp = Annotations.loadMask(hypMask, 1)
    hyp.saveTsv(outFile)
