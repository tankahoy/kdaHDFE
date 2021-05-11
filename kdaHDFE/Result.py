from scipy.stats import t
import pandas as pd
import numpy as np


class Result:
    def __init__(self, results, std_error, covariant_matrix):
        """
        Adjusted results of regression

        :param results: The mostly unadjusted results from OLS bar the degrees of freedom that was adjusted for clusters
        :type results: statsmodels.regression.linear_model.RegressionResults
        """
        self.raw_results = results

        self.covariant_matrix = covariant_matrix
        self.obs = self.raw_results.nobs
        self.df = self.raw_results.df_resid
        self.r_sq = self.raw_results.rsquared
        self.r_sq_adj = 1 - (self.obs - 1) / self.df * (1 - self.r_sq)

        self.params = self.raw_results.params
        self.bse = std_error
        self.t_values = self.params / self.bse
        self.p_values = pd.Series(2 * t.sf(np.abs(self.t_values), self.df), index=list(self.params.index))
        self.resid = self.raw_results.resid

    def results_out(self, variable_list):
        # todo: We want this to act akin to the regression results part in SRGWAS
        raise NotImplementedError
