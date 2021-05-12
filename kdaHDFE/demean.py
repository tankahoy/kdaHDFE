import numpy as np


def demean(variables, df, fixed_effects, obs, epsilon=1e-8, max_iter=1e6, mean_squared_error=10):
    """
    Using large numbers of fixed effects can slow programs down. This uses demeaning of groupby to reduce the
    complexity of the operation

    :return: Demeaned data frame
    :rtype: pd.DataFrame
    """
    demean_return = df.copy()
    for covariant in variables:
        demeaned = df.copy()
        mse = mean_squared_error

        iter_count = 0
        demeans_cache = np.zeros(obs, np.float64)

        while mse > epsilon:
            for fe in fixed_effects:
                demeaned[covariant] = demeaned[covariant] - demeaned.groupby(fe)[covariant].transform('mean')

            iter_count += 1
            mse = np.linalg.norm(demeaned[covariant].values - demeans_cache)
            demeans_cache = demeaned[covariant].copy().values

            if iter_count > max_iter:
                raise RuntimeWarning(f'MSE fails to converge to epsilon within {max_iter} iterations')

        demean_return[[covariant]] = demeaned[[covariant]]
    return demean_return
