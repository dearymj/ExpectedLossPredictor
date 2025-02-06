# GUI related
import tkinter as tk
from tkinter import filedialog, messagebox

# Data Processing
import pandas as pd
import numpy as np

# Machine Learning Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

# ------------------------------
# Global variables (for demo)
# ------------------------------
df = None
X_train, X_test = None, None
y_train, y_test = None, None
logreg, dtree = None, None

RECOVERY_RATE = 0.10
LGD = 1 - RECOVERY_RATE  # Loss Given Default


# ------------------------------
# Functions
# ------------------------------
def append_text(text_widget, msg):
    """
    Utility function to append a message to a Text widget,
    then scroll to the end.
    """
    text_widget.config(state=tk.NORMAL)
    text_widget.insert(tk.END, msg + "\n")
    text_widget.see(tk.END)
    text_widget.config(state=tk.DISABLED)


def load_data():
    """
    Load the CSV file and display the first 5 rows in the text widget.
    """
    global df

    filepath = filedialog.askopenfilename(
        title='Select CSV File',
        filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
    )
    if not filepath:
        return  # User canceled

    try:
        local_df = pd.read_csv(filepath)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read CSV:\n{e}")
        return

    # Store it as global df
    df = local_df.dropna()  # simplistic drop of NaNs for the example

    # Print the first 5 rows
    text_output.delete('1.0', tk.END)
    append_text(text_output, f"Loaded data from: {filepath}")
    append_text(text_output, "First 5 rows:")
    append_text(text_output, str(df.head()))


def train_models():
    """
    Train Logistic Regression and Decision Tree on the dataset.
    """
    global df, X_train, X_test, y_train, y_test, logreg, dtree

    if df is None:
        messagebox.showwarning("Warning", "Please load the data first.")
        return

    # Check required columns
    required_cols = {
        "credit_lines_outstanding",
        "loan_amt_outstanding",
        "total_debt_outstanding",
        "income",
        "years_employed",
        "fico_score",
        "default"
    }
    if not required_cols.issubset(set(df.columns)):
        messagebox.showerror(
            "Error",
            f"Data must contain columns: {required_cols}"
        )
        return

    # Drop customer_id if present (not needed)
    if 'customer_id' in df.columns:
        df.drop(columns=['customer_id'], inplace=True)

    # Prepare features and target
    y = df['default']
    X = df.drop(columns=['default'])

    # Simple train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Logistic Regression
    logreg = LogisticRegression(max_iter=1000, solver='liblinear')
    logreg.fit(X_train, y_train)

    # Train Decision Tree
    dtree = DecisionTreeClassifier(max_depth=5, random_state=42)
    dtree.fit(X_train, y_train)

    # Evaluate (AUC)
    pd_logreg_test = logreg.predict_proba(X_test)[:, 1]
    pd_dtree_test = dtree.predict_proba(X_test)[:, 1]

    auc_logreg = roc_auc_score(y_test, pd_logreg_test)
    auc_dtree = roc_auc_score(y_test, pd_dtree_test)

    # Print to GUI
    text_output.delete('1.0', tk.END)
    append_text(text_output, "Models trained successfully!")
    append_text(text_output, f"Logistic Regression AUC: {auc_logreg:.3f}")
    append_text(text_output, f"Decision Tree AUC:       {auc_dtree:.3f}")


def expected_loss(model, loan_features):
    """
    Given a trained model and the features of a new loan
    (as a single-row DataFrame), returns the expected loss =
    PD * LGD * loan_amt_outstanding.
    """
    pd_pred = model.predict_proba(loan_features)[:, 1]
    loan_amount = loan_features['loan_amt_outstanding'].values
    el = pd_pred * LGD * loan_amount
    return el


def predict_random_samples():
    """
    Pick 3 random samples from the test set, show their features,
    actual default, and expected loss from each model.
    """
    global X_test, y_test, logreg, dtree

    if X_test is None or y_test is None or logreg is None or dtree is None:
        messagebox.showwarning("Warning", "Please train the models first.")
        return

    text_output.delete('1.0', tk.END)

    sample_indices = np.random.choice(X_test.index, 3, replace=False)
    sample_loans = X_test.loc[sample_indices]

    # LogReg predictions
    sample_el_logreg = expected_loss(logreg, sample_loans)

    # Decision Tree predictions
    sample_el_dtree = expected_loss(dtree, sample_loans)

    for i, idx in enumerate(sample_indices):
        append_text(text_output, f"---- Loan Sample (index={idx}) ----")
        append_text(text_output, str(sample_loans.loc[idx]))
        actual_default = y_test.loc[idx]
        append_text(text_output, f"Default Actual: {actual_default}")
        append_text(text_output,
                    f"Expected Loss (LogReg):  {sample_el_logreg[i]:.2f}")
        append_text(text_output,
                    f"Expected Loss (DTree):   {sample_el_dtree[i]:.2f}")
        append_text(text_output, "")


def exit_app():
    root.destroy()


# ------------------------------
# Tkinter GUI Setup
# ------------------------------
root = tk.Tk()
root.title("Expected Loss GUI | By MJ Yuan")

# Main frame
frame = tk.Frame(root, padx=10, pady=10)
frame.pack()

# Buttons
btn_load = tk.Button(frame, text="1. Load Data", width=20, command=load_data)
btn_load.grid(row=0, column=0, padx=5, pady=5)

btn_train = tk.Button(frame, text="2. Train Models", width=20, command=train_models)
btn_train.grid(row=0, column=1, padx=5, pady=5)

btn_predict = tk.Button(frame, text="3. Predict Samples", width=20, command=predict_random_samples)
btn_predict.grid(row=0, column=2, padx=5, pady=5)

btn_exit = tk.Button(frame, text="Exit", width=20, command=exit_app)
btn_exit.grid(row=0, column=3, padx=5, pady=5)

# Text widget for output
text_output = tk.Text(frame, width=100, height=20, state=tk.DISABLED)
text_output.grid(row=1, column=0, columnspan=4, padx=5, pady=5)

# Start the GUI event loop
root.mainloop()
