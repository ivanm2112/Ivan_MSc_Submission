import os

start_time = int(os.getenv("start_time"))
end_time = int(os.getenv("end_time"))
number_of_steps = int(os.getenv("number_of_steps"))
step_size = int(os.getenv("step_size"))

num_iterations = int(os.getenv("num_iterations"))
initial_collect_steps = int(os.getenv("initial_collect_steps"))
collect_steps_per_iteration = int(os.getenv("collect_steps_per_iteration"))
replay_buffer_max_length = int(os.getenv("replay_buffer_max_length"))

batch_size = int(os.getenv("batch_size"))
learning_rate = float(os.getenv("learning_rate"))
log_interval = int(os.getenv("log_interval"))

num_eval_episodes = int(os.getenv("num_eval_episodes"))
eval_interval = int(os.getenv("eval_interval"))

fc_layer_params = tuple(map(int, os.getenv("fc_layer_params").split(',')))


