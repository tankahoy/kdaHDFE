from kdaHDFE import formula_transform


class HDFE:
    def __init__(self, formula):
        self.phenotype, self.covariants, self.fixed_effects, self.clusters = formula_transform(formula)
