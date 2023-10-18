#### Free energy extracting
# pylint: disable = C0103, C0114, C0116, C0301, R0914

def last_column_average_computing(data_lines):
    """Define the function to compute the average of the last value in each line."""
    total = 0
    for line in data_lines:
        values = line.split()       # Split the line into individual values
        total += float(values[-1])  # Add the last value to the total
        # print(line)
    return total / len(data_lines)  # Return the average
