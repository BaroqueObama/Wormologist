class HyperParams:
    def __init__(self, lambda_cen=1.0, lambda_rad=1.0, lambda_off=2.0, dcen_threshold=20.0, drad_threshold=1.0, min_number_assignments=-1, ensure_lap=True, only_hungarian=False, c0=150):
        self.lambda_cen = lambda_cen
        self.lambda_rad = lambda_rad
        self.lambda_off = lambda_off
        self.dcen_threshold = dcen_threshold
        self.drad_threshold = drad_threshold
        self.min_number_assignments = min_number_assignments
        self.ensure_lap = ensure_lap
        self.only_hungarian = only_hungarian
        self.c0 = c0

    def __str__(self):
        return f"HyperParams(lambda_cen={self.lambda_cen}, lambda_rad={self.lambda_rad}, lambda_off={self.lambda_off}, dcen_threshold={self.dcen_threshold}, drad_threshold={self.drad_threshold}, min_number_assignments={self.min_number_assignments}, c0={self.c0})"