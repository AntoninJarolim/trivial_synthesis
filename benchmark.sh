#!/bin/bash

# Extract the array name from the command-line argument
benchmark_name=$1

# Set trivial synthesis (one-by-one) or AR (paynt)
script_name="./trivial_synthesis/main.py"
method_name="one"
[ "$2" == "AR" ] && script_name="./synthesis/paynt.py" && method_name="AR"


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
  properties=("first line\nsecond line"
    "third line\nfourth line"
    "fifth line")
  ;;
"array2")
  properties=(
    "line 1\nline 2\nline 3"
    "line 4\nline 5\nline 6"
    "line 7")
  ;;
*)
  echo "Invalid benchmark name"
  exit 1
  ;;
esac

# save properties file
old_properties_file=$(cat $folder"sketch.props")
mkdir -p "log/${benchmark_name}/${method_name}"

for prop in "${properties[@]}"; do
  # Change the contents of the file to the current array prop
  echo "BENCHMARK: running property "$prop
  echo -e "$prop" >$folder"sketch.props"

  # Run the Python script and redirect output to file
  prop_names=$(echo "$prop" | tr '\n' '-' | sed 's/-$//')

  python $script_name --project $folder >"log/${benchmark_name}/${method_name}/${prop_names}.log"
done

echo -e $old_properties_file >$folder"sketch.props"
