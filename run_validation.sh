 #!/bin/bash

## Define path for workspaces (needed to run reasoner and metacontrol_sim in different ws)
## You need to create a "config.sh file in the same folder defining your values for these variables"
source config.sh
export METACONTROL_WS_PATH
export REASONER_WS_PATH
export PYTHON3_VENV_PATH
export MODEL_TRAINING_PATH
export SIMULATION_TESTS_PATH

####
#  Experiment parameters
####
declare -a configs=("teb_v0_a0_b0" "dwa_v0_a0_b0")


declare exp="validation"
declare x_goal="20"
declare y_goal="-11"
declare yaw_goal="-1.55"

declare record=1

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

start_simulation () {
	echo ""
	echo "Start a new simulation"
	echo ""
	echo "Launch roscore"
	gnome-terminal --window --geometry=80x24+10+10 -- bash -c "source $METACONTROL_WS_PATH/devel/setup.bash; roscore; exit"

	sleep 3

	echo "Launching: Simulation tests world.launch"
	gnome-terminal --window --geometry=80x24+10+10 -- bash -c "source $SIMULATION_WS_PATH/devel/setup.bash;
	roslaunch simulation_tests world.launch world:=validation.world;
	exit"

	sleep 8
}

start_observers () {
	echo "Launch and load observers"
	gnome-terminal --window --geometry=80x24+10+10 -- bash -c "source $METACONTROL_WS_PATH/devel/setup.bash;
	rosrun rosgraph_monitor monitor;
	exit"

	sleep 2

	bash -c  "
		rosservice call /load_observer \"name: 'SafetyObserverTrain'\";
		rosservice call /load_observer \"name: 'NarrownessObserverTrain'\";
		rosservice call /load_observer \"name: 'ObstacleDensityObserverTrain'\";
		rosservice call /load_observer \"name: 'PerformanceObserverDWATrain'\";
		rosservice call /load_observer \"name: 'PerformanceObserverTEBTrain'\";
		exit;"
}

restart_simulation () {
	# Check that there are not running ros nodes
	kill_running_ros_nodes
	# If gazebo is running, it may take a while to end
	wait_for_gzserver_to_end

	start_simulation

	start_observers
}

echo "Make sure there is no other gazebo instances or ROS nodes running:"
# Check that there are not running ros nodes
kill_running_ros_nodes
# If gazebo is running, it may take a while to end
wait_for_gzserver_to_end

start_simulation

start_observers


declare fail=0

for run in 1 ; do
	for config in ${configs[@]} ; do
		fail=2
		while [ $fail -gt 0 ] ; do
			fail=2

			# echo "Launching: Spawn boxer"
			gnome-terminal --window --geometry=80x24+10+10 -- bash -c "source $SIMULATION_WS_PATH/devel/setup.bash;
			roslaunch simulation_tests spawn_boxer.launch;
			exit"

			# echo "Launching: move_base"
			# echo "Configuration: $config"
			gnome-terminal --window --geometry=80x24+10+10 -- bash -c "source $METACONTROL_WS_PATH/devel/setup.bash;
			roslaunch $config $config.launch;
			exit;"

			start_observers

			sleep 1

			# Rosbag
			if [ $record -eq 1 ] ; then
				# timeout 120s gnome-terminal --window --geometry=80x24+10+10 -- bash -c  "
				gnome-terminal --window --geometry=80x24+10+10 -- bash -c  "
				echo 'Start record';
				cd $MODEL_TRAINING_PATH/bags;
				rosbag record -e '/metrics/.*' -O exp${exp}_c${config}_w${w}_C${C}_r${run}.bag;
				exit;"
			fi

			# echo "Start navigation manager"
			# timeout 120s bash -c "
			bash -c "
			echo 'Starting navigation';
			cd $MODEL_TRAINING_PATH/scripts;
			./navigation_manager.py $x_goal $y_goal $yaw_goal;
			exit;"
			fail=$?
			if [ $fail -eq 1 ]; then
				echo "${fail}: Failed run, try again"
			fi
			if [ $fail -gt 1 ]; then
				echo "${fail}: Failed navigation, Restarting"
				restart_simulation
			fi 


			# Stop record node
			gnome-terminal --window --geometry=80x24+10+10 -- bash -c "
			rosnode list | grep record* | xargs rosnode kill;
			rosnode list | grep graph_monitor | xargs rosnode kill; 
			exit;"

			# echo "Kill robot and move_base"
			gnome-terminal --window --geometry=80x24+10+10 -- bash -c "
			rosnode kill move_base slam_gmapping ekf_localization robot_state_publisher controller_spawner;
			rosservice call gazebo/delete_model '{model_name: /}';
			exit"
			sleep 1
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
