import pandas as pd

def remove_rows_containing_strings(file_path, column_name, strings_file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Load the list of strings to find from a CSV file
    strings_df = pd.read_csv(strings_file_path)
    strings_to_find = strings_df['node_ID'].tolist()

    # Iterate over each string in the list of strings to find
    for string in strings_to_find:
        # Find all rows in the specified column that contain the current string
        # The ~ operator inverts the result, so this selects all rows that do NOT contain the string
        # This updates the DataFrame to only include the rows that do not contain the string
        df = df[~df[column_name].str.contains(string)]

    # Save the updated DataFrame back to the CSV file
    # The index=False argument prevents pandas from writing row indices to the CSV file
    df.to_csv(file_path, index=False)

strings_file_path = 'virtual_node.csv'  # replace with the path to your CSV file containing the list of strings to find
remove_rows_containing_strings('testing_data.csv', 'node_ID', strings_file_path)
