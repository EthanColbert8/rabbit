from rabbit.mappings.ratio import Asymmetry

class Steerability(Asymmetry):
    """
    Steerability is derived from the forward-backward asymmetry of the cos_phiPrime observable,
    so this is a simple mapping to extract it from a fit.
    """
    def init(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_flat(self, params, observables):
        return 4.0 * super().compute_flat(params, observables)