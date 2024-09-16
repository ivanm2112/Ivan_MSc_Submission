import os

start_time=int(os.getenv("start_time"))
end_time=int(os.getenv("end_time"))
number_of_steps=int(os.getenv("number_of_steps"))
step_size=int(os.getenv("step_size"))

# Hyperparameters
initial_num_iterations=int(os.getenv("initial_num_iterations"))
#sub_num_iterations=int(os.getenv("sub_num_iterations"))

initial_collect_steps=int(os.getenv("initial_collect_steps"))
collect_steps_per_iteration=int(os.getenv("collect_steps_per_iteration"))
replay_buffer_capacity=int(os.getenv("replay_buffer_capacity"))

batch_size=int(os.getenv("batch_size"))

critic_learning_rate=float(os.getenv("critic_learning_rate"))
actor_learning_rate=float(os.getenv("actor_learning_rate"))
alpha_learning_rate=float(os.getenv("alpha_learning_rate"))
target_update_tau=float(os.getenv("target_update_tau"))
target_update_period=int(os.getenv("target_update_period"))
gamma=float(os.getenv("gamma"))
reward_scale_factor=float(os.getenv("reward_scale_factor"))

actor_fc_layer_params=tuple(map(int, os.getenv("actor_fc_layer_params").split(',')))
critic_joint_fc_layer_params=tuple(map(int, os.getenv("critic_joint_fc_layer_params").split(',')))

log_interval=int(os.getenv("log_interval"))

num_eval_episodes=int(os.getenv("num_eval_episodes"))
eval_interval=int(os.getenv("eval_interval"))

policy_save_interval=int(os.getenv("policy_save_interval"))
