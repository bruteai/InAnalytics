import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.style as style

# Use a Seaborn style without grid by default
plt.style.use("seaborn-v0_8-white")

# Increase default font size for better readability
plt.rcParams.update({'font.size': 10})

# Set the number of rows for the synthetic dataset (e.g., 1000 rows)
n = 1000

# Generate Customer IDs (e.g., CUST0001, CUST0002, ...)
customer_ids = [f"CUST{i:04d}" for i in range(1, n+1)]

# Generate random ages between 18 and 90
ages = np.random.randint(18, 90, size=n)

# Randomly assign Gender
genders = np.random.choice(["Male", "Female"], size=n)

# Randomly assign Canadian provinces
provinces_list = [
    "Ontario", "Quebec", "British Columbia", "Alberta", "Manitoba",
    "Saskatchewan", "Nova Scotia", "New Brunswick",
    "Newfoundland and Labrador", "Prince Edward Island"
]
provinces = np.random.choice(provinces_list, size=n)

# Generate Policy Premiums between $300 and $2000
policy_premiums = np.random.uniform(300, 2000, size=n).round(2)

# Generate number of claims between 0 and 5
num_claims = np.random.randint(0, 6, size=n)

# Generate Frequency (number of transactions) between 1 and 12
frequency = np.random.randint(1, 13, size=n)

# Generate Recency (days since last transaction) between 1 and 365
recency = np.random.randint(1, 366, size=n)

# Generate Monetary values (total spend) between $500 and $20000
monetary = np.random.uniform(500, 20000, size=n).round(2)

# Generate a synthetic Historical CLV:
# Here, we simulate it as Policy_Premium * Frequency plus some random noise.
historical_clv = (policy_premiums * frequency + np.random.uniform(-100, 100, size=n)).round(2)

# Define some sample customer feedback options.
feedback_options = [
    "Excellent service", "Very satisfied", "Satisfied",
    "Neutral", "Unsatisfied", "Poor service"
]
customer_feedback = [random.choice(feedback_options) for _ in range(n)]

# Create the DataFrame
df = pd.DataFrame({
    "CustomerID": customer_ids,
    "Age": ages,
    "Gender": genders,
    "Province": provinces,
    "Policy_Premium": policy_premiums,
    "Num_Claims": num_claims,
    "Frequency": frequency,
    "Recency": recency,
    "Monetary": monetary,
    "Historical_CLV": historical_clv,
    "Customer_Feedback": customer_feedback
})

# Write the DataFrame to a CSV file
csv_filename = "insurance_customers.csv"
df.to_csv(csv_filename, index=False)
print(f"Synthetic dataset generated and saved as '{csv_filename}'.")

# --- Generate a Summary Image ---
fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

# Subplot 1: Histogram of Ages
axs[0, 0].hist(df["Age"], bins=20, color='#6ec6ff', edgecolor='black')
axs[0, 0].set_title("Age Distribution", fontsize=14)
axs[0, 0].set_xlabel("Age", fontsize=12)
axs[0, 0].set_ylabel("Count", fontsize=12)

# Subplot 2: Bar Plot of Province Counts
province_counts = df["Province"].value_counts()
axs[0, 1].bar(province_counts.index, province_counts.values, color='#8df98d', edgecolor='black')
axs[0, 1].set_title("Province Distribution", fontsize=14)
axs[0, 1].set_xlabel("Province", fontsize=12)
axs[0, 1].set_ylabel("Count", fontsize=12)
axs[0, 1].tick_params(axis='x', labelrotation=45, labelsize=10)
for label in axs[0, 1].get_xticklabels():
    label.set_ha('right')

# Subplot 3: Histogram of Policy Premiums
axs[1, 0].hist(df["Policy_Premium"], bins=20, color='#ff9999', edgecolor='black')
axs[1, 0].set_title("Policy Premium Distribution", fontsize=14)
axs[1, 0].set_xlabel("Policy Premium ($)", fontsize=12)
axs[1, 0].set_ylabel("Count", fontsize=12)

# Subplot 4: Scatter Plot - Policy Premium vs. Historical CLV
axs[1, 1].scatter(df["Policy_Premium"], df["Historical_CLV"], alpha=0.6, color='#9a60d1')
axs[1, 1].set_title("Policy Premium vs. Historical CLV", fontsize=14)
axs[1, 1].set_xlabel("Policy Premium ($)", fontsize=12)
axs[1, 1].set_ylabel("Historical CLV ($)", fontsize=12)

# Turn off the grid and remove top/right spines for all subplots
for ax in axs.ravel():
    ax.grid(False)  # Disable background grid lines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

image_filename = "dataset_summary.png"
plt.savefig(image_filename, dpi=300)
plt.close()

print(f"Summary image generated and saved as '{image_filename}'.")
