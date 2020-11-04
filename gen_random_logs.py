from glob import glob
import os
import subprocess

NUM_RAND_PER_TRACE = 2  # 1000
trace_files = glob('trace/*.gz')[:2]
for trace_file in trace_files:
	cmd = "./{} -warmup_instructions 1000000 -simulation_instructions 10000000 -traces {}"
	trace_name = os.path.basename(trace_file).split('.')[0]
	store_fldr = os.path.join('random_logs', '{}', trace_name)
	store_fldr_c1 = store_fldr.format('rand-config1')
	if not os.path.exists(store_fldr_c1):
		os.makedirs(store_fldr_c1)
	store_fldr_c2 = store_fldr.format('rand-config2')
	if not os.path.exists(store_fldr_c2):
		os.makedirs(store_fldr_c2)

	for i in range(NUM_RAND_PER_TRACE):
		out_file = open("{}/{}.txt".format(store_fldr_c1, i), 'w')
		new_cmd = cmd.format('rand-config1', trace_file)

		# run the command
		print('Running : ', new_cmd.split())
		subprocess.Popen(new_cmd.split(), stdout=out_file, stderr=out_file)

		new_cmd = cmd.format('rand-config2', trace_file)
		# run the command
		print('Running : ', new_cmd.split())
		out_file = open("{}/{}.txt".format(store_fldr_c2, i), 'w')
		subprocess.Popen(new_cmd.split(), stdout=out_file, stderr=out_file)
