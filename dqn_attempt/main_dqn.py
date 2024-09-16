import os
import subprocess
import sys
import time

from dotenv import load_dotenv

def main():
    
    # Get user inputs
    initial_value = 1440
    x = 30

    # if initial_value/x % 2 != 0:
    #     print("The increments need to work back to midnight.")
    #     SystemExit

    current_input_value = initial_value 
    initial_num_iterations = int(os.getenv("num_iterations"))
    saving_path = os.getenv("chosen_path_saving")
    # process = subprocess.Popen(['python3','train_dqn.py', str(current_input_value), str(initial_num_iterations), saving_path])
    # process.wait()
    for i in range(48):
        # Calculate the input value for this iteration
        current_input_value -= x
        num_iterations = initial_num_iterations 
        #if i == 0 else sub_num_iterations
        # Open shell and execute the command
        process = subprocess.Popen(['python3','train_dqn.py', str(current_input_value), str(num_iterations),saving_path])
        process.wait()

if __name__ == "__main__":
    load_dotenv(sys.argv[1])
    main()
