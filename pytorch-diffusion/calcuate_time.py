import matplotlib.pyplot as plt

def calculate_time(file_path):
    training_times = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            if 'Valid' not in line and '[1]' not in line:
                training_times.append(float(line[-6:]))
            
    return training_times

# File path to the log file
file_path = "log_2000_bfloat16_precision.txt"  # Replace with your actual file path

# Parse log file and extract losses
times = calculate_time(file_path)
times.sort()
med_time = times[len(times) // 2] # time (ms) scaled per image
print(med_time)
med_time *= 50000 # time (ms) per epoch
med_time /= 1000 # time (s) per epoch
med_time *= 500 # time (s) total
med_time /= 60 # time (min) total
print(med_time)
med_time /= 60 # time (hr) total
print(med_time)
