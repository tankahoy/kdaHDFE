from kdaHDFE import formula_transform

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

        :return:
        """

        # Failed formula's to test that AssertionError can catch them
        formula_list = ["CoV", "BMI+Drinking~CoV", "rs012~", "rs012~Cov+Cov2|DFE|CL|Random"]

        for formula in formula_list:
            with self.subTest():
                self.assertRaises(AssertionError, lambda: formula_transform(formula)[:1])




if __name__ == '__main__':
    unittest.main()
