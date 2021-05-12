from kdaHDFE import HDFE
import pandas as pd
import time


if __name__ == '__main__':
    runs = 500

    df = pd.read_csv(r"C:\Users\Samuel\PycharmProjects\kdaHDFE\Data\ExampleData.csv")
    formula = 'rs012~BMI+Gender'

    # Demean the common variables
    hdfe = HDFE(df, formula)
    demeaned = hdfe.demean(hdfe.covariants)

    print(demeaned)

    # Original test
    start = time.time()
    for i in range(runs):
        HDFE(df, formula).reg_hdfe()
    print(f"Took {time.time() - start} for Original")

    formula = 'rs012~BMI|Gender'

    start = time.time()
    for i in range(runs):
        HDFE(df, formula).reg_hdfe()
    print(f"Took {time.time() - start} for Demean per")

    start = time.time()
    for i in range(runs):
        # Assign the phenotype
        demeaned["rs012"] = df["rs012"]
        HDFE(demeaned, formula).reg_hdfe(1)
    print(f"Took {time.time() - start} for Demeaning Once")
