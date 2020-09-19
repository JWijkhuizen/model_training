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
#  Default values, set if no parameters are given
####

# declare -a configs=("teb_v0_a0_b0" "teb_v1_a0_b0" "dwa_v0_a0_b0" "dwa_v1_a0_b0")
declare -a configs=("dwa_v0_a0_b0" "teb_v0_a0_b0")

declare -a ws=(3 4)
declare -a Cs=(8 6)
declare l=21

declare n_max=0
declare n=0
for w in "${ws[@]}" ; do
	for C in "${Cs[@]}" ; do
		n=$(( w*l/C ))
		((n > n_max)) && n_max=$n
	done
done

declare exp="test1"
declare sx=100
declare x_goal=46
# declare -a runs = 

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
	roslaunch simulation_tests world.launch;
	exit"

	echo "Spawn corridor and obstacles"
	bash -c "
	cd $MODEL_TRAINING_PATH/scripts;
	./simrun_init_environment.py ${ws[0]} $l $n_max 0 0 1;
	exit;"
	echo "Spawn corridor and obstacles"
	bash -c "
	cd $MODEL_TRAINING_PATH/scripts;
	./simrun_init_environment.py ${ws[1]} $l $n_max 28 100 1;
	exit;"

	echo "Launch and load observers"
	# gnome-terminal --window --geometry=80x24+10+10 -- bash -c "source $METACONTROL_WS_PATH/devel/setup.bash;
	# rosrun rosgraph_monitor monitor;
	# exit"
	gnome-terminal --window --geometry=80x24+10+10 -- bash -c "
	roslaunch simulation_tests observers.launch;
	exit"	
}
restart_simulation () {
	# Check that there are not running ros nodes
	kill_running_ros_nodes
	# If gazebo is running, it may take a while to end
	wait_for_gzserver_to_end

	start_simulation
	
	bash -c "
	cd $MODEL_TRAINING_PATH/scripts;
	./simrun_move_environment.py $w $l $C $sx $n_max;
	exit;"
}

echo "Make sure there is no other gazebo instances or ROS nodes running:"
# Check that there are not running ros nodes
kill_running_ros_nodes
# If gazebo is running, it may take a while to end
wait_for_gzserver_to_end

start_simulation

declare fail=0
# declare moved_obs=0

# for w in "${ws[@]}" ; do

sleep 1
# for C in "${Cs[@]}" ; do
for run in 0 1 ; do
	# moved_obs=0
	# echo "Environment width = ${w}, C = ${C}, run = ${run}"
	timeout 80s bash -c "
	cd $MODEL_TRAINING_PATH/scripts;
	./simrun_move_obstacles.py 3 $l 8 $sx $n_max 0 0;
	./simrun_move_obstacles.py 4 $l 6 $sx $n_max 21 101;
	exit;"
	# moved_obs=$?
	# if [ $moved_obs -eq 0 ]; then
	# 	echo "Failed to move obstacles, Restarting"
	# 	restart_simulation
	# fi 
	echo "Finished moving environment"
	for config in ${configs[@]} ; do
		fail=2
		while [ $fail -gt 0 ] ; do
			fail=2
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
				timeout 120s gnome-terminal --window --geometry=80x24+10+10 -- bash -c  "
				echo 'Start record';
				cd $MODEL_TRAINING_PATH/bags;
				rosbag record -e '/metrics/.*' -O exp${exp}_c${config}_r${run}.bag;
				exit;"
			fi

			# echo "Start navigation manager"
			echo "Fail is ${fail}"
			timeout 120s bash -c "
			echo 'Starting navigation';
			cd $MODEL_TRAINING_PATH/scripts;
			./navigation_manager.py $x_goal 0 0;
			exit;"
			fail=$?
			echo "Fail is ${fail}"
			if [ $fail -eq 1 ]; then
				echo "Failed run, try again"
			fi
			# if [ $fail -gt 1 ]; then
			# 	echo "Failed navigation, Restarting"
			# 	restart_simulation
			# fi 


			# Stop record node
			gnome-terminal --window --geometry=80x24+10+10 -- bash -c "rosnode list | grep record* | xargs rosnode kill; exit;"

			# echo "Kill robot and move_base"
			gnome-terminal --window --geometry=80x24+10+10 -- bash -c "
			rosnode kill move_base slam_gmapping ekf_localization robot_state_publisher controller_spawner;
			rosservice call gazebo/delete_model '{model_name: /}';
			exit"
			sleep 1
		done
	done
	sx=$((sx+1))
done


echo "Experiments finished!!"


# # Check that there are not running ros nodes
kill_running_ros_nodes
# Wait for gazebo to end
# wait_for_gzserver_to_end
