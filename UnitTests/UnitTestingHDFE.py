from kdaHDFE import form_transfer

import unittest


class UnitTestingHDFE(unittest.TestCase):

    def test_formula(self):
        """
        Test various forms of formula's to ensure they have been picked up by raise or complete successfully

        Failed Operations to catch
        ---------------------------
        Cov: No phenotype
        BMI+Drinking~Cov: More than 1 phenotype

        :return:
        """

        # Failed formula's to test that AssertionError can catch them
        formula_list = ["CoV", "BMI+Drinking~CoV"]

        for formula in formula_list:
            with self.subTest():
                self.assertRaises(AssertionError, lambda: form_transfer(formula)[:1])




if __name__ == '__main__':
    unittest.main()
