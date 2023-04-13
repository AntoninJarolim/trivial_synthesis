#!/bin/bash

# Extract the array name from the command-line argument
benchmark_name=$1

# Set trivial synthesis (one-by-one) or AR (paynt)
script_name="./trivial_synthesis/main.py"
method_name="trivial"
[ "$2" == "AR" ] && script_name="./synthesis/paynt.py" && method_name="AR"

if [ -z "$3" ]; then
  memory=1
else
  memory="$3"
fi

case $benchmark_name in
"robot-battery-stay-LRA")
  folder="./models/pomdp/no_goal_state/robot-battery-stay/"
  properties=(
    'LRA < 0.01 [ "exploring" ]'
    'LRA < 0.08 [ "exploring" ]'
    'LRA < 0.15 [ "exploring" ]'
    'LRA < 0.2  [ "exploring" ]'
    'LRA < 0.25 [ "exploring" ]'

    'LRA > 0.01 [ "exploring" ]'
    'LRA > 0.08 [ "exploring" ]'
    'LRA > 0.15 [ "exploring" ]'
    'LRA > 0.2  [ "exploring" ]'
    'LRA > 0.25 [ "exploring" ]'
  )
  ;;
"robot-battery-stay-P")
  properties=(""
    ""
    "")
  ;;
"robot-battery-LRA")
  folder="./models/pomdp/no_goal_state/robot-battery/"
  properties=(
    'LRAmax=? [ "exploring" ]'
    'LRA>0.16 [ "exploring" ]'
    'LRA>0.17 [ "exploring" ]'
    'LRA>0.20 [ "exploring" ]'
    'LRA>0.22 [ "exploring" ]'
    'LRA>0.25 [ "exploring" ]'
  )
  ;;
*)
  echo "Invalid benchmark name"
  exit 1
  ;;
esac

# save properties file

old_properties_file=$(cat $folder"sketch.props")
write_back_old_file() {
  echo -e $old_properties_file >$folder"sketch.props"
  exit 1
}
trap 'write_back_old_file' SIGINT

mkdir -p "log/${benchmark_name}/${method_name}"

for prop in "${properties[@]}"; do
  # Change the contents of the file to the current array prop
  echo "BENCHMARK: running property "$prop
  echo -e "$prop" >$folder"sketch.props"

  # Run the Python script and redirect output to file
  prop_names=$(echo "$prop" | tr '\n' '-' | sed 's/-$//')

  timeout 600 \
    python $script_name --project $folder --pomdp-memory-size $memory \
    > "log/${benchmark_name}/${method_name}/${prop_names}.log"
done

write_back_old_file

