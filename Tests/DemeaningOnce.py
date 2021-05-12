from kdaHDFE import HDFE, cal_df, formula_transform, demean

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

        phenotype, covariant, fixed_effects, clusters = formula_transform(formula)
        demeaned = demean(covariant + phenotype, df, fixed_effects, len(df))
        rank = cal_df(df, fixed_effects)

        time_new = []
        for i in range(runs):
            start = time.time()
            HDFE(demeaned, formula).reg_hdfe(rank, False)
            time_new.append(time.time() - start)
        print(np.average(time_new))
        print("")

        run_rows.append([name, np.average(time_new)])

    write_csv(Path(__file__).parent, "TimingUpdate", ["Model Name", "New"], run_rows)

