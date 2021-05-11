from kdaHDFE import formula_transform, cal_df
import statsmodels.api as sm
import pandas as pd
import numpy as np


class HDFE:
    def __init__(self, data_frame, formula, epsilon=1e-8, max_iter=1e6, mean_squared_error=10):

        # Extract variable names from formula
        self.phenotype, self.covariants, self.fixed_effects, self.clusters = formula_transform(formula)

        # Setup the database reference
        self.df = data_frame
        self.obs = len(self.df)

        # Some standard variables to be used for demeaning / clustering
        self.mean_squared_error = mean_squared_error
        self.epsilon = epsilon
        self.max_iter = max_iter

    def demean_data_frame(self):
        """
        Using large numbers of fixed effects can slow programs down. This uses demeaning of groupby to reduce the
        complexity of the operation

        :return: Demeaned data frame
        :rtype: pd.DataFrame
        """
        demean_return = self.df.copy()

        for covariant in self.covariants + self.phenotype:
            demeaned = self.df.copy()
            mse = self.mean_squared_error

            iter_count = 0
            demeans_cache = np.zeros(self.obs, np.float64)
            while mse > self.epsilon:
                for fe in self.fixed_effects:
                    demeaned[covariant] = demeaned[covariant] - demeaned.groupby(fe)[covariant].transform('mean')

                iter_count += 1
                mse = np.linalg.norm(demeaned[covariant].values - demeans_cache)
                demeans_cache = demeaned[covariant].copy().values
                if iter_count > self.max_iter:
                    raise RuntimeWarning(f'MSE fails to converge to epsilon within {self.max_iter} iterations')

            demean_return[[covariant]] = demeaned[[covariant]]
        return demean_return

    def reg_hdfe(self):

        # Demean the data and determine the rank from the absorbed fixed effects from demeaning
        demeaned, rank = self._reg_demean()

        # Calculate the base unadjusted OLS results, add residuals to result for clustering
        result = sm.OLS(demeaned[self.phenotype], demeaned[self.covariants]).fit()
        demeaned['resid'] = result.resid

    def _reg_demean(self):
        """
        Certain model specifications may require use to add an intercept such as when there is no need to demean as
        there are no fixed effects. If we have fixed effects, demean the data and calculate degrees of freedome lost
        via construction of demeaned data.

        :return: Demeaned DataFrame, rank of degrees of freedom
        :rtype: (pd.DataFrame, int)
        """
        # Add a constant if the model lacks any covariants / fe or lacks both covariants and fe but not clusters
        if len(self.covariants) == 0 or len(self.fixed_effects) == 0 or \
                (len(self.covariants) == 0 and len(self.fixed_effects) == 0 and len(self.clusters) > 0):

            demeaned = self.df.copy()
            demeaned["Const"] = [1.0 for _ in range(len(demeaned))]
            self.covariants = self.covariants + ["Const"]
            return demeaned, 0

        else:
            return self.demean_data_frame(), cal_df(self.df, self.fixed_effects)
