import os
import subprocess
import sys
import time

from dotenv import load_dotenv

def main():
    """_summary_
    """
    # Get user inputs
    initial_value = 1020
    x = 30

    # if initial_value/x % 2 != 0:
    #     print("The increments need to work back to midnight.")
    #     SystemExit

    current_input_value = initial_value 
    initial_num_iterations = int(os.getenv("initial_num_iterations"))
    sub_num_iterations = int(os.getenv('sub_num_iterations'))
    learning_path = os.getenv("chosen_path_learning")
    learning_path = f"{learning_path}/{str(current_input_value)}"
    saving_path = os.getenv("chosen_path_saving")
    process = subprocess.Popen(['python3','train_sac.py', str(current_input_value), str(initial_num_iterations), learning_path, saving_path])
    process.wait()
    # for i in range(48):
    #     learning_path = ""
    #     # Calculate the input value for this iteration
    #     current_input_value -= x
    #     num_iterations = initial_num_iterations
    #     learning_path = os.getenv("chosen_path_learning")
    #     learning_path = f"{learning_path}/{str(current_input_value)}"
    #     # Open shell and execute the command
    #     process = subprocess.Popen(['python3','train_sac_better.py', str(current_input_value), str(num_iterations), learning_path, saving_path])
    #     process.wait()


if __name__ == "__main__":
    load_dotenv(sys.argv[1])
    main()
