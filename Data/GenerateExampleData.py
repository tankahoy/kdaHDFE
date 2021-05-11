"""
Create a dummy set of data to mimic UK Biobank data of IID, Example snp dosage 1-5, Some Covariant
"""
from random import randint, uniform
from miscSupports import flip_list
from csvObject import write_csv
from pathlib import Path

if __name__ == '__main__':

    # Variables that alter the dimensions of the array but not its content
    sample_size = 100000
    random_snps = 5

    # IID
    iid = [i for i in range(1, sample_size + 1)]

    # BMI
    bmi = [uniform(14.5, 40.5) for _ in range(sample_size)]

    # Place of Birth ID
    pob = [randint(0, 5000) for _ in range(sample_size)]

    # Age
    yob = [randint(65, 90) for _ in range(sample_size)]

    # Gender
    gender = [randint(0, 1) for _ in range(sample_size)]

    # Ever Smoked
    smoke = [randint(0, 1) for _ in range(sample_size)]

    # Units drank per week
    alcohol = [randint(0, 30) for _ in range(sample_size)]

    # Average daily intake of calories
    calories = [randint(1000, 3500) for _ in range(sample_size)]

    # asthmatic
    asthma = [randint(0, 1) for _ in range(sample_size)]

    # Ever used drugs
    drug_user = [randint(0, 1) for _ in range(sample_size)]

    # 5 random snp dosages
    dosages = [[randint(0, 2) for i in range(sample_size)] for _ in range(random_snps)]

    out_rows = [iid, bmi, pob, yob, gender, smoke, alcohol, calories, asthma, drug_user] + dosages
    headers = ["IID", "BMI", "PoB", "YoB", "Gender", "Smoke", "Alcohol", "Calories", "Asthma", "Drug_user"] + \
              [f"rs{i}{i+1}{i+2}" for i in range(random_snps)]
    write_csv(Path(__file__).parent, "ExampleData", headers, flip_list(out_rows))
