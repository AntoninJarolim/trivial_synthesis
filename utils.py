import argparse
import os


def read_props(props_path):
    file = open(props_path)
    props = []
    for line in file:
        line = line.strip()
        if not line.startswith("//") and len(line) > 0:
            props.append(line)
    return props


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_path")
    args = parser.parse_args()
    template_path = os.path.join(args.project_path, "sketch.templ")
    props_path = os.path.join(args.project_path, "sketch.props")
    props = read_props(props_path)
    return template_path, props


def print_mdp(mdp_model):
    print(mdp_model.transition_matrix)
    for state in mdp_model.states:
        if state.id in mdp_model.initial_states:
            print("State {} is initial".format(state.id))
        for action in state.actions:
            for transition in action.transitions:
                print(f"From state {state} by action {action}, "
                      f"with probability {transition.value()}, "
                      f"go to state {transition.column}")
