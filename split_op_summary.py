
import pandas as pd
import os

# Load the data
csv_file = r'D:\learning\modle_Tan\op_summary.csv'
output_excel = 'IR×FER.xlsx'

# Sheet names in the requested order (RunNum 1 to 18)
sheet_names = [
    '225N+NO IR', '225N+WCFC%', '225N+FULL IR',
    '-10%N+NO IR', '-10%N+WCFC%', '-10%N+FULL IR',
    '-20%N+NO IR', '-20%N+WCFC%', '-20%N+FULL IR',
    '-30%N+NO IR', '-30%N+WCFC%', '-30%N+FULL IR',
    '-40%N+NO IR', '-40%N+WCFC%', '-40%N+FULL IR',
    '-50%N+NO IR', '-50%N+WCFC%', '-50%N+FULL IR'
]

if not os.path.exists(csv_file):
    print(f"Error: {csv_file} not found.")
else:
    df = pd.read_csv(csv_file)
    
    # Check if RUNNUM and WRR14 exist
    if 'RUNNUM' not in df.columns or 'WRR14' not in df.columns:
        print("Error: RUNNUM or WRR14 column not found in op_summary.csv")
    else:
        # Get all unique RUNNUMs and sort them
        runnums = sorted(df['RUNNUM'].unique())
        
        # Create ExcelWriter
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            for idx, runnum in enumerate(runnums):
                # Filter data for current runnum
                subset = df[df['RUNNUM'] == runnum].copy()
                
                # Calculate the mean of WRR14
                mean_wrr14 = subset['WRR14'].mean()
                
                # Add a row at the end with the mean
                # We'll create a new row as a Series and append it
                mean_row = {col: "" for col in df.columns}
                mean_row['YEAR'] = "Mean"
                mean_row['WRR14'] = mean_wrr14
                
                # Convert mean_row to a DataFrame and concat it
                mean_df = pd.DataFrame([mean_row])
                final_df = pd.concat([subset, mean_df], ignore_index=True)
                
                # Determine sheet name
                if idx < len(sheet_names):
                    current_sheet_name = sheet_names[idx]
                else:
                    current_sheet_name = f"Sheet{runnum}"
                
                # Write to sheet
                final_df.to_excel(writer, sheet_name=current_sheet_name, index=False)
                print(f"Written sheet for runnum {runnum} as '{current_sheet_name}'")
        
        print(f"Successfully saved {output_excel} with {len(runnums)} sheets.")
