"""KS detector: two-sample Kolmogorov-Smirnov test per CONTINUOUS feature."""
from scipy.stats import ks_2samp
from ._common import aggregate


def ks_detector(profile, batch):
    per = {f: float(ks_2samp(profile.ref[f].values, batch[f].values).statistic)
           for f in profile.continuous}
    return aggregate(per, "ks"), per
