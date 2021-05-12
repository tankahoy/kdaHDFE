from kdaHDFE import formula_transform, cal_df, clustered_error, is_nested, robust_err, Result

import statsmodels.api as sm
import pandas as pd
import numpy as np


class HDFE:
    def __init__(self, data_frame, formula, robust=False, cm="cgm", ps_def=True, epsilon=1e-8, max_iter=1e6,
                 mean_squared_error=10):

        # Extract variable names from formula
        self.phenotype, self.covariants, self.fixed_effects, self.clusters = formula_transform(formula)

        # Setup the database reference
        self.df = data_frame
        self.obs = len(self.df)

        # Some standard variables to be used for demeaning / clustering
        self.mean_squared_error = mean_squared_error
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.robust = robust
        # todo: IF clusters > 1 then cgm2 is preferred
        self.cm = cm
        self.ps_def = ps_def

    def demean(self, variables):
        """
        Using large numbers of fixed effects can slow programs down. This uses demeaning of groupby to reduce the
        complexity of the operation

        :return: Demeaned data frame
        :rtype: pd.DataFrame
        """
        demean_return = self.df.copy()
        for covariant in variables:
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
        return demean_return[variables + self.fixed_effects]

    def reg_hdfe(self, demean_option=0):
        """
        Run a demeaned version of the data frame via absorption of FE's and adjust results relative to the demeaned data

        :param demean_option: The demeaning operation to perform.
            If 0, demeans phenotype + covariants. Useful for single regressions that will handle all the data complexity
            for you.
            If 1, demeans phenotype, assumes that you have already demeaned covariant's
        :type demean_option: int

        :return: Results of the regresstion
        :rtype: Result
        """

        demeaned, rank = self._reg_demean(demean_option)
        print(rank)

        # Calculate the base unadjusted OLS results, add residuals to result for clustering and update degrees of
        # freedom from demeaning
        # todo Allow for missing
        # todo write unit test of missing data to test that data with missing still runs
        result = sm.OLS(demeaned[self.phenotype], demeaned[self.covariants]).fit()
        demeaned['resid'] = result.resid  # Ever used?
        result.df_resid = result.df_resid - rank

        std_error, covariant_matrix = self._reg_std(result, rank, demeaned)

        return Result(result, std_error, covariant_matrix)

    def _reg_demean(self, demean_option):
        """
        Certain model specifications may require use to add an intercept such as when there is no need to demean as
        there are no fixed effects. If we have fixed effects, demean the data and calculate degrees of freedom lost
        via construction of demeaned data.

        :param demean_option: The demeaning operation to perform.
            If 0, demeans phenotype + covariants. Useful for single regressions that will handle all the data complexity
            for you.
            If 1, demeans phenotype, assumes that you have already demeaned covariant's
        :type demean_option: int

        :return: Demeaned DataFrame, rank of degrees of freedom
        :rtype: (pd.DataFrame, int)
        """
        # Add a constant if the model lacks any covariants / fe or lacks both covariants and fe but not clusters
        if len(self.covariants) == 0 or len(self.fixed_effects) == 0 or \
                (len(self.covariants) == 0 and len(self.fixed_effects) == 0 and len(self.clusters) > 0):

            # Demean == DataFrame
            demeaned = self.df.copy()
            demeaned["Const"] = [1.0 for _ in range(len(demeaned))]
            self.covariants = self.covariants + ["Const"]
            return demeaned, 0

        else:
            if demean_option == 0:
                # Demean the whole dataframe and calculate the degrees of freedom after demeaning
                # todo also here
                return self.demean(self.phenotype + self.covariants), cal_df(self.df, self.fixed_effects)

            elif demean_option == 1:
                # Demean the phenotype
                phenotype = self.demean(self.phenotype)
                demeaned = self.df.copy()
                demeaned[self.phenotype] = phenotype[self.phenotype]

                # Return this after calculating degrees of freedom
                # todo: Cal_df should always be the same though right? Why are we doing this per run when it can be
                #  per model?
                # todo: Have a -recalculate_df- property or something to update the df if the formula is changed.
                return demeaned, cal_df(self.df, self.fixed_effects)

            else:
                raise IndexError(f"Demean_option takes the value of 0 for demeaning all variables in formula or 1 for"
                                 f"demeaning phenotype")

    def _reg_std(self, result, rank, demeaned_df):
        """
        If we have clusters, we need to cluster the standard error depending on the clustering method of self.cm

        Otherwise, we need to create robust or non robust standard errors from the standard errors calculated adjusted
        for de-meaning

        :param result: OLS result
        :param rank: rank of degrees of freedom
        :param demeaned_df: Demeaned Database for clustering if required
        :return: The standard error and the covariance matrix
        """
        # Now we need to update the standard errors of the OLS based on robust and clustering
        if (len(self.clusters) == 0) & (self.robust is False):
            std_error = result.bse * np.sqrt((result.nobs - len(self.covariants)) / (result.nobs - len(self.covariants)
                                                                                     - rank))
            covariance_matrix = result.normalized_cov_params * result.scale * result.df_resid / result.df_resid

        elif (len(self.clusters) == 0) & (self.robust is True):
            covariance_matrix = robust_err(demeaned_df, self.covariants, result.nobs, len(self.covariants), rank)
            std_error = np.sqrt(np.diag(covariance_matrix))

        else:
            nested = is_nested(demeaned_df, self.fixed_effects, self.clusters, self.covariants)

            covariance_matrix = clustered_error(demeaned_df, self.covariants, self.clusters, result.nobs,
                                                len(self.covariants), rank, nested=nested, c_method=self.cm,
                                                psdef=self.ps_def)
            std_error = np.sqrt(np.diag(covariance_matrix))

        return std_error, covariance_matrix
