import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_insurance_analysis_csv(output_file="insurance_analysis.csv", n=1000):
    """
    Generates a synthetic dataset for insurance analysis with columns:
      - Policy_Type
      - Region
      - Claim_Amount
      - Policy_Premium
      - Transaction_Date

    Args:
        output_file (str): Name of the CSV file to write.
        n (int): Number of rows to generate.
    """

    # Define some example categories
    policy_types = ["Auto", "Home", "Life", "Health", "Travel", "Commercial"]
    regions = ["Ontario", "Quebec", "British Columbia", "Alberta", "Manitoba", "Saskatchewan"]

    # Date range: let's pick from Jan 1, 2023 to Dec 31, 2023
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range_days = (end_date - start_date).days

    # Arrays to store data
    data_policy_type = []
    data_region = []
    data_claim_amount = []
    data_policy_premium = []
    data_transaction_date = []

    for _ in range(n):
        # Randomly choose a policy type & region
        p_type = random.choice(policy_types)
        region = random.choice(regions)

        # Claim amount: random between $0 and $20,000, with some chance for large outliers
        # We'll add some skew by squaring a uniform random
        claim_amt = np.round((np.random.rand() ** 2) * 20000, 2)

        # Policy premium: random between $300 and $2,000
        policy_prem = np.round(np.random.uniform(300, 2000), 2)

        # Transaction date: random day between start and end
        offset_days = np.random.randint(0, date_range_days + 1)
        t_date = start_date + timedelta(days=offset_days)

        # Append to arrays
        data_policy_type.append(p_type)
        data_region.append(region)
        data_claim_amount.append(claim_amt)
        data_policy_premium.append(policy_prem)
        data_transaction_date.append(t_date.strftime("%Y-%m-%d"))

    # Create a DataFrame
    df = pd.DataFrame({
        "Policy_Type": data_policy_type,
        "Region": data_region,
        "Claim_Amount": data_claim_amount,
        "Policy_Premium": data_policy_premium,
        "Transaction_Date": data_transaction_date
    })

    # Write to CSV
    df.to_csv(output_file, index=False)
    print(f"Synthetic insurance analysis dataset generated: {output_file}")

if __name__ == "__main__":
    generate_insurance_analysis_csv()
