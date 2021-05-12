from kdaHDFE.legacy import ols_high_d_category
from kdaHDFE import HDFE, cal_df, formula_transform

from csvObject import write_csv
from pathlib import Path
import pandas as pd
import numpy as np
import time

if __name__ == '__main__':

    df = pd.read_csv(r"C:\Users\Samuel\PycharmProjects\kdaHDFE\Data\ExampleData.csv")
    print(df.columns)

    formula_list = ['rs012~BMI', 'rs012~BMI+Gender', 'rs012~BMI+Gender+Smoke+Alcohol+Calories+Asthma+Drug_user',
                    'rs012~BMI+Gender|PoB', 'rs012~BMI+Gender|PoB|PoB', 'rs012~BMI+Gender|PoB+YoB',
                    'rs012~BMI+Gender|PoB+YoB+Asthma', 'rs012~BMI+Gender|PoB|PoB+YoB',
                    'rs012~BMI+Gender|PoB+YoB|PoB+YoB']

    row_names = ["Cov", "2Cov", "7Cov", "2Cov+FE", "2Cov+FE+Cl", "2Cov+2FE", "2Cov+3FE", "2Cov+FE+2CL", "2Cov+2FE+2CL"]

    runs = 100
    run_rows = []
    for name, formula in zip(row_names, formula_list):
        print(formula)
        print("OLD")
        time_old = []
        for i in range(runs):
            start = time.time()
            results = ols_high_d_category(df, formula=formula)
            time_old.append(time.time() - start)
        print(np.average(time_old))

        phenotype, covariant, fixed_effects, clusters = formula_transform(formula)
        rank = cal_df(df, fixed_effects)

        print("NEW")
        time_new = []
        for i in range(runs):
            start = time.time()
            HDFE(df, formula).reg_hdfe(rank)
            time_new.append(time.time() - start)

        run_rows.append([name, np.average(time_old), np.average(time_new)])
        print(np.average(time_new))

    print(run_rows)

    write_csv(Path(__file__).parent, "Timing", ["Model Name", "Old", "New"], run_rows)
