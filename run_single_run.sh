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

# declare -a configs=("dwa1" "dwa2" "teb1" "teb2")
declare -a configs=("teb_v0_a0_b0" "teb_v1_a0_b0" "dwa_v0_a0_b0" "dwa_v1_a0_b0")

declare -a ws=(4 3)
declare -a Cs=(4 8)
declare l=21

declare n_max=0
declare n=0
for w in "${ws[@]}" ; do
	for C in "${Cs[@]}" ; do
		n=$(( w*l/C ))
		((n > n_max)) && n_max=$n
	done
done

declare exp="n1"
declare sx=1
declare x_goal=28

declare record=1

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

#printf "Argument init_position is %s\n" "$init_position"
#printf "Argument goal_position is %s\n" "$goal_position"
#printf "Argument nav_profile is %s\n" "$nav_profile"
#printf "Argument launch reconfiguration is %s\n" "$launch_reconfiguration"
#printf "Argument obstacles is %s\n" "$obstacles"
#printf "Argument increase power is %s\n" "$increase_power"

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


echo "Make sure there is no other gazebo instances or ROS nodes running:"

# Check that there are not running ros nodes
kill_running_ros_nodes
# If gazebo is running, it may take a while to end
wait_for_gzserver_to_end

# Get x and y initial position from yaml file - takes some creativity :)
# declare init_pos_x=$(cat $METACONTROL_WS_PATH/src/metacontrol_experiments/yaml/initial_positions.yaml | grep S$init_position -A 5 | tail -n 1 | cut -c 10-)
# declare init_pos_y=$(cat $METACONTROL_WS_PATH/src/metacontrol_experiments/yaml/initial_positions.yaml | grep S$init_position -A 6 | tail -n 1 | cut -c 10-)

# cat $METACONTROL_WS_PATH/src/metacontrol_experiments/yaml/goal_positions.yaml | grep G$goal_position -A 12 | tail -n 12 > $METACONTROL_WS_PATH/src/metacontrol_sim/yaml/goal.yaml

echo ""
echo "Start a new simulation - Goal position: $goal_position - Initial position  $init_position - Navigation profile: $nav_profile"
echo ""
echo "Launch roscore"
gnome-terminal --window --geometry=80x24+10+10 -- bash -c "source $METACONTROL_WS_PATH/devel/setup.bash; roscore; exit"
#Sleep Needed to allow other launchers to recognize the roscore

sleep 3

echo "Launching: Simulation tests world.launch"
gnome-terminal --window --geometry=80x24+10+10 -- bash -c "source $SIMULATION_WS_PATH/devel/setup.bash;
roslaunch simulation_tests world.launch;
exit"

echo "Spawn corridor and obstacles"
bash -c "
cd $MODEL_TRAINING_PATH/scripts;
./simrun_init_environment.py ${ws[0]} $l $n_max;
exit;"

echo "Launch and load observers"
gnome-terminal --window --geometry=80x24+10+10 -- bash -c "source $METACONTROL_WS_PATH/devel/setup.bash;
rosrun rosgraph_monitor monitor;
exit"

sleep 1

bash -c  "
rosservice call /load_observer \"name: 'SafetyObserverTrain'\";
rosservice call /load_observer \"name: 'NarrownessObserverTrain'\";
rosservice call /load_observer \"name: 'ObstacleDensityObserverTrain'\";
exit;"

for w in "${ws[@]}" ; do
	for C in "${Cs[@]}" ; do
		for run in 0 1 2 3 4 5 6 7 8 9 10 ; do
			echo "Environment width = ${w}, C = ${C}, run = ${run}"
			bash -c "
			cd $MODEL_TRAINING_PATH/scripts;
			./simrun_move_environment.py $w $l $C $sx $n_max;
			exit;"
			for config in ${configs[@]} ; do
				# echo "Launching: Spawn boxer"
				gnome-terminal --window --geometry=80x24+10+10 -- bash -c "source $SIMULATION_WS_PATH/devel/setup.bash;
				roslaunch simulation_tests spawn_boxer.launch;
				exit"

				# echo "Launching: move_base"
				echo "Configuration: $config"
				gnome-terminal --window --geometry=80x24+10+10 -- bash -c "source $METACONTROL_WS_PATH/devel/setup.bash;
				roslaunch $config $config.launch;
				exit"

				# Rosbag
				if [ $record -eq 1 ] ; then
					gnome-terminal --window --geometry=80x24+10+10 -- bash -c  "
					cd $MODEL_TRAINING_PATH/bags;
					rosbag record -e '/metrics/.*' -O exp${exp}_c${config}_w${w}_C${C}_r${run}.bag;
					exit;"
				fi

				# echo "Start navigation manager"
				bash -c "
				cd $MODEL_TRAINING_PATH/scripts;
				./navigation_manager.py $x_goal 0 0;
				exit;"

				# Stop record node
				gnome-terminal --window --geometry=80x24+10+10 -- bash -c "rosnode list | grep record* | xargs rosnode kill; exit;"

				# echo "Kill robot and move_base"
				gnome-terminal --window --geometry=80x24+10+10 -- bash -c "
				rosnode kill move_base slam_gmapping ekf_localization robot_state_publisher controller_spawner;
				rosservice call gazebo/delete_model '{model_name: /}';
				exit"

				sleep 1
			done
			sx=$((sx+1))
		done
	done
done

echo "Experiments finished!!"
# echo "Running log and stop simulation node"
# bash -ic "source $METACONTROL_WS_PATH/devel/setup.bash;
# roslaunch metacontrol_experiments stop_simulation.launch obstacles:=$obstacles goal_nr:=$goal_position increase_power:=$increase_power record_bags:=$record_rosbags;
# exit "
# echo "Simulation Finished!!"

# Check that there are not running ros nodes
kill_running_ros_nodes
# Wait for gazebo to end
# wait_for_gzserver_to_end
