from kdaHDFE import formula_transform, HDFE, cal_df

from pathlib import Path
import pandas as pd
import numpy as np
import unittest


class UnitTestingHDFE(unittest.TestCase):

    def test_formula(self):
        """
        Test various forms of formula's to ensure they have been picked up by raise or complete successfully

        Failed Operations to catch
        ---------------------------
        Cov: No phenotype
        BMI+Drinking~Cov: More than 1 phenotype
        rs012~: Phenotype with no right hand side
        rs012~Cov+Cov2|DFE|CL|Random: More than three sections of formula

        Types of operations to run
        --------------------------
        bmi~Cov: phenotype ~ Covariant
        bmi~Cov+Cov2|DFE: phenotype ~ Covariant | Fixed effect
        bmi~Cov|DFE+DFE2|DFE: phenotype ~ Covariant | Fixed effect | Cluster
        bmi~Cov||DFE+DFE2: phenotype ~ Covariant | | Cluster
        bmi~|DFE|DFE: phenotype ~ | Fixed effect | Cluster
        """

        # Failed formula's to test that AssertionError can catch them
        formula_list = ["CoV", "BMI+Drinking~CoV", "rs012~", "rs012~Cov+Cov2|DFE|CL|Random"]
        for formula in formula_list:
            with self.subTest():
                self.assertRaises(AssertionError, lambda: formula_transform(formula)[:1])

        # Valid formula and then length or args they should return
        formula_list = ["bmi~Cov", "bmi~Cov+Cov2|DFE", "bmi~Cov|DFE+DFE2|DFE", "bmi~Cov||DFE=DFE2", "bmi~|DFE|DFE"]
        expected_lengths = [[1, 1, 0, 0], [1, 2, 1, 0], [1, 1, 2, 1], [1, 1, 0, 1], [1, 0, 1, 1]]
        for formula, validate in zip(formula_list, expected_lengths):
            with self.subTest():
                self.assertEqual([len(sub_list) for sub_list in formula_transform(formula)], validate)

    @staticmethod
    def _example_data():
        """Load the example data"""
        return pd.read_csv(Path(Path(__file__).parent.parent, "Data", "ExampleData.csv"))

    def test_nan(self):
        """
        Data may be missing, we need to make sure we are handling it
        """

        # Load the example database, then change all 0 to NaN
        df = self._example_data()
        df = df.replace(0, np.NaN)

        # Test the regression still runs
        formula = "BMI~Gender+Smoke+Alcohol"
        r = HDFE(df, formula).reg_hdfe(cal_df(df, formula_transform(formula)[2]))
        self.assertEqual(int(r.params["Gender"]), 9)

    def test_invalid_names(self):
        """
        Individuals may place a variable that does not exist in the DataFrame, we need to catch this.
        """

        df = self._example_data()

        formula = "bmi~Geder"
        rank = cal_df(df, formula_transform(formula)[2])
        self.assertRaises(AssertionError, lambda: HDFE(df, formula).reg_hdfe(rank)[:1])


if __name__ == '__main__':
    unittest.main()
