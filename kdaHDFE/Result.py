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

    def conf_interval(self, conf=0.05):
        """Calculate the 95% confidence interval"""
        conf_min = self.params - t.ppf(1 - conf / 2, self.df) * self.bse
        conf_max = self.params + t.ppf(1 - conf / 2, self.df) * self.bse
        return conf_min, conf_max

    def results_out(self, variable_list):
        """
        Returns for each variable in the list of variables

        [Parameters, standard error, p values, obs, 95%min CI, 95%max CI]

        :param variable_list: A list of variables, or a string if you only want a single variable
        :type variable_list: list | str

        :return: A list of lists, where each list are the results in float
        :rtype:list[list[float, float, float, float, float, float]]
        """
        results = []

        # If only one variable is called recast variable_list as a list
        if isinstance(variable_list, str):
            variable_list = [variable_list]

        for v in variable_list:
            results.append([self.params[v], self.bse[v], self.p_values[v], self.obs] +
                           [c[v] for c in self.conf_interval()])
        return results
