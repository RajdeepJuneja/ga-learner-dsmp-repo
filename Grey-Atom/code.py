# --------------
# code ends here

loan_groupby= banks.groupby(['Loan_Status'])['ApplicantIncome', 'Credit_History']

mean_values =loan_groupby.mean()


# code ends here


# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 

bank = pd.read_csv(path)
categorical_var = bank.select_dtypes(include = 'object')
print(categorical_var)
numerical_var = bank.select_dtypes(include = 'number')
print(numerical_var)
 

# code starts here






# code ends here


# --------------
# code starts here

banks = bank.drop(['Loan_ID'],axis = 1)
print(banks.isnull().sum())
bank_mode = banks.mode()
banks.fillna(bank_mode.iloc[0],inplace = True)
print(banks)
#code ends here


# --------------
# Code starts here
avg_loan_amount = pd.pivot_table(banks, index = ['Gender', 'Married', 'Self_Employed'] ,values ='LoanAmount')



# code ends here



# --------------
# code starts here
loan_approved_se = banks.loc[(banks['Self_Employed'] == 'Yes' ) &(banks['Loan_Status'] == 'Y'),['Loan_Status']].count()

loan_approved_nse= banks.loc[(banks['Self_Employed'] == 'No' ) &(banks['Loan_Status'] == 'Y'),['Loan_Status']].count()

percentage_se = (loan_approved_se * 100/ 614)
percentage_se = percentage_se[0]

# code ends here
percentage_nse = (loan_approved_nse * 100 /614)
percentage_nse = percentage_nse[0]


# --------------
# code starts here
#loan_term =lambda Loan_Amount_Term : int(Loan_Amount_Term) / 12

loan_term = banks['Loan_Amount_Term'].apply(lambda x: int(x)/12 )


big_loan_term=len(loan_term[loan_term>=25])

print(big_loan_term)



# code ends here


