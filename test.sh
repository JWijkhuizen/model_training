 #!/bin/bash

 usage()
 {
	echo "Usage: $0 -i <init_position: (1 / 2 / 3)>"
	echo "          -g <goal_position: (1 / 2 / 3)>"
	echo "          -n <nav_profile: ('fast' / 'standard' / 'safe' or fX_vX_rX)>"
	echo "          -r <reconfiguration: (true / false)>"
	echo "          -o <obstacles: (0 / 1 / 2 / 3)>"
	echo "          -p <increase_power: (0/1.1/1.2/1.3)>"
	echo "          -b <record rosbags: ('true' / 'false')>"
	echo "          -e <nfr energy threshold : ([0 - 1])>"
	echo "          -s <nfr safety threshold : ([0 - 1])>"
	echo "          -c <close reasoner terminal : ('true' / 'false')>"
	exit 1
 }
## Define path for workspaces (needed to run reasoner and metacontrol_sim in different ws)
## You need to create a "config.sh file in the same folder defining your values for these variables"
source config.sh
export METACONTROL_WS_PATH
export REASONER_WS_PATH
export PYTHON3_VENV_PATH
export MODEL_TRAINING_PATH
export SIMULATION_TESTS_PATH

####
#  Default values, set if no parameters are given
####

## Define initial navigation profile
# Possible values ("fast" "standard" "safe")
# also any of the fx_vX_rX metacontrol configurations
declare config="dwa_v1_a0_b0"

## Define initial position
# Possible values (1, 2, 3)
declare init_pos_x=0
declare init_pos_y=0


## Define goal position
# Possible values (1, 2, 3)
declare goal_pos_x=31
declare goal_pos_y=0

if [ "$1" == "-h" ]
then
	usage
    exit 0
fi

while getopts ":i:g:n:r:o:p:e:s:c:" opt; do
  case $opt in
    i) init_position="$OPTARG"
    ;;
    g) goal_position="$OPTARG"
    ;;
    n) nav_profile="$OPTARG"
    ;;
    r) launch_reconfiguration="$OPTARG"
    ;;
    o) obstacles="$OPTARG"
    ;;
    p) increase_power="$OPTARG"
    ;;
    e) nfr_energy="$OPTARG"
    ;;
    s) nfr_safety="$OPTARG"
    ;;
	c) close_reasoner_terminal="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    	usage
    ;;
  esac
done

wait_for_gzserver_to_end () {

	for t in $(seq 1 100)
	do
		if test -z "$(ps aux | grep gzserver | grep -v grep )"
		then
			# echo " -- gzserver not running"
			break
		else
			echo " -- gzserver still running"
		fi
		sleep 1
	done
}

kill_running_ros_nodes () {
	# Kill all ros nodes that may be running
	for i in $(ps aux | grep ros | grep -v grep | awk '{print $2}')
	do
		echo "kill -2 $i"
		kill -2 $i;
	done
	sleep 1
}

kill_record () {
	# Kill all ros nodes that may be running
	for i in $(ps aux | grep ros | grep -v grep | grep record | awk '{print $2}')
	do
		echo "kill -2 $i"
		# kill -2 $i;
	done
	# sleep 1
}

echo "Test 1 2 3 .... Test 1 2 3 ....."

echo "start"
timeout 2s bash -c "sleep 10"
echo "stop"

declare fail=1
bash -c "
cd $MODEL_TRAINING_PATH/scripts;
./simrun_test.py 1;
exit;"
fail=$?
echo $fail
while [ $fail -eq 1 ] ; do
	echo $fail
	bash -c "
	cd $MODEL_TRAINING_PATH/scripts;
	./simrun_test.py 0;
	exit;"
	fail=$?
	echo $fail
	sleep 1
done


# echo "Launching: move_base"
# echo "Configuration: $config"
# gnome-terminal --window --geometry=80x24+10+10 -- bash -c "source $METACONTROL_WS_PATH/devel/setup.bash;
# roslaunch $config $config.launch;
# read -rsn 1 -p 'Press any key to close this terminal...'"

# echo "Running log and stop simulation node"
# bash -ic "source $METACONTROL_WS_PATH/devel/setup.bash;
# roslaunch metacontrol_experiments stop_simulation.launch obstacles:=$obstacles goal_nr:=$goal_position increase_power:=$increase_power record_bags:=$record_rosbags;
# exit "
# echo "Simulation Finished!!"

# Check that there are not running ros nodes
# kill_running_ros_nodes
# Wait for gazebo to end
# wait_for_gzserver_to_end
