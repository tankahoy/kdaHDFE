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

        Types of operations to run
        --------------------------
        bmi~Cov: phenotype ~ Covariant
        bmi~Cov+Cov2|DFE: phenotype ~ Covariant | Fixed effect
        bmi~Cov|DFE+DFE2|DFE: phenotype ~ Covariant | Fixed effect | Cluster
        bmi~Cov||DFE+DFE2: phenotype ~ Covariant | | Cluster
        bmi~|DFE|DFE: phenotype ~ | Fixed effect | Cluster

        :return: None
        :rtype: None
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


if __name__ == '__main__':
    unittest.main()
