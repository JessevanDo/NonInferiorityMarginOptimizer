import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_1samp
from scipy.optimize import root_scalar

def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv")])
    return file_path

def prompt_for_value(prompt, default_value):
    root = tk.Tk()
    root.withdraw()
    value = simpledialog.askfloat("Input", prompt, initialvalue=default_value)
    return value

def calculate_statistics(data):
    mean = data.mean()
    std = data.std()
    median = data.median()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    return mean, std, median, q1, q3, iqr

def perform_non_inferiority_test_parametric(data, reference_mean, non_inferiority_margin):
    threshold = reference_mean - non_inferiority_margin
    test_statistic, p_value = ttest_1samp(data, threshold, alternative='greater')
    return test_statistic, p_value

def find_non_inferiority_margin(data, reference_value, test_function):
    def objective_function(margin):
        _, p_value = test_function(data, reference_value, margin)
        return p_value - 0.05

    # Print diagnostic information for a range of margins
    for margin in [0, 0.25, 0.5, 0.75, 1.0]:
        print(f"Objective function at {margin}: {objective_function(margin)}")

    # Check if the function changes signs in the interval [0, 1]
    f_start = objective_function(0)
    f_end = objective_function(1)
    if f_start * f_end > 0:
        raise ValueError("Function does not change signs within the interval [0, 1]")

    result = root_scalar(objective_function, bracket=[0, 1], method='brentq')
    return result.root

def plot_data(data, reference_value, non_inferiority_margin, metric='Median'):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, alpha=0.5, label='Test Sample')
    plt.axvline(reference_value, color='r', linestyle='dashed', linewidth=2, label=f'Reference {metric}')
    plt.axvline(reference_value - non_inferiority_margin, color='g', linestyle='dashed', linewidth=2,
                label='Non-Inferiority Threshold')
    plt.xlabel('Dice Similarity Coefficient')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title(f'Distribution of Dice Similarity Coefficient with {metric} Non-Inferiority')
    plt.show()

def main():
    # Step 1: Select CSV file
    file_path = select_file()
    if not file_path:
        messagebox.showinfo("Information", "No file selected. Exiting.")
        return

    # Load the data
    df = pd.read_csv(file_path)

    # Check if 'Dice' column exists
    if 'Dice' not in df.columns:
        messagebox.showerror("Error", "'Dice' column not found in the selected file. Exiting.")
        return

    # Extract 'Dice' column data
    dice_data = df['Dice'].dropna()

    # Step 2: Calculate statistics
    mean, std, median, q1, q3, iqr = calculate_statistics(dice_data)

    # Step 3: Prompt user for reference median and reference mean
    reference_median = prompt_for_value("Enter the reference median value:", median)
    reference_mean = prompt_for_value("Enter the reference mean value:", mean)

    # Step 4: Find non-inferiority margins for a p-value of 0.05
    try:
        non_inferiority_margin_p = find_non_inferiority_margin(dice_data, reference_mean, perform_non_inferiority_test_parametric)
        non_inferiority_percentage_p = (non_inferiority_margin_p / reference_mean) * 100
    except ValueError as e:
        non_inferiority_margin_p = None
        non_inferiority_percentage_p = None
        print(f"Failed to find non-inferiority margin for parametric test: {e}")

    # Plot data for parametric test
    if non_inferiority_margin_p is not None:
        plot_data(dice_data, reference_mean, non_inferiority_margin_p, metric='Mean')

    # Perform non-inferiority tests
    if non_inferiority_margin_p is not None:
        test_statistic_p, p_value_p = perform_non_inferiority_test_parametric(dice_data, reference_mean,
                                                                              non_inferiority_margin_p)
    else:
        test_statistic_p, p_value_p = None, None

    # Display results
    result_message = (
        f"Calculated Mean: {mean}\n"
        f"Calculated SD: {std}\n"
        f"Calculated Median: {median}\n"
        f"Calculated IQR: {iqr}\n\n"
        f"Reference Median: {reference_median}\n"
        f"Parametric Test:\n"
        f"Reference Mean: {reference_mean}\n"
        f"Non-Inferiority Margin: {non_inferiority_margin_p} ({non_inferiority_percentage_p:.2f}% of mean)\n"
        f"Test Statistic: {test_statistic_p}\n"
        f"P-Value: {p_value_p}\n"
        f"Non-Inferiority Test Result: {'Pass' if p_value_p is not None and p_value_p < 0.05 else 'Fail'}"
    )

    messagebox.showinfo("Non-Inferiority Test Result", result_message)
    print(result_message)

if __name__ == "__main__":
    main()
