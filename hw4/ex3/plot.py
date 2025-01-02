import matplotlib.pyplot as plt

# Initialize lists to store steps and errors
steps = []
errors = []

# Read data from output.txt
with open('output.log', 'r') as file:
    for line in file:
        # Skip empty lines
        line = line.strip()
        if not line:
            continue
            
        # Split and validate line
        parts = line.split()
        if len(parts) < 2:
            print(f"Warning: Skipping invalid line: {line}")
            continue
            
        try:
            step = int(parts[0])
            error = float(parts[1])
            steps.append(step)
            errors.append(error)
        except (ValueError, IndexError) as e:
            print(f"Error processing line: {line}")
            print(f"Error details: {e}")

# Plot if we have data
if steps and errors:
    plt.plot(steps, errors, marker='o')
    plt.xlabel('Steps')
    plt.ylabel('Error')
    plt.title('DimX = 1024')
    plt.grid(True)
    plt.show()
else:
    print("No valid data found to plot")