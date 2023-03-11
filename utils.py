import argparse
import os
import re

import stormpy


def read_props(props_path):
    file = open(props_path)
    props = []
    for line in file:
        line = line.strip()
        if not line.startswith("//") and len(line) > 0:
            props.append(line)
    print("Specification: ", end="")
    print(", ".join(props))
    return props


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_path")
    parser.add_argument("--memory-size", default=1, type=int)
    args = parser.parse_args()
    template_path = os.path.join(args.project_path, "sketch.templ")
    props_path = os.path.join(args.project_path, "sketch.props")
    props = read_props(props_path)
    return template_path, props, args.memory_size


def exact_specifications(specification: str):
    return re.sub(r"([a-zA-Z\s]+)(.+)(\[)", r"\1=?\3", specification)


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


def read_pomdp_drn(sketch_path):
    explicit_model = None
    try:
        builder_options = stormpy.core.DirectEncodingParserOptions()
        builder_options.build_choice_labels = True
        explicit_model = stormpy.core._build_sparse_model_from_drn(
            sketch_path, builder_options)
    except:
        raise ValueError('Failed to read sketch file in a .drn format')
    return explicit_model


def print_to_drn(model):
    stormpy.export_to_drn(model, "model.drn")


def get_all_attrs(obj) -> dict:
    dic = {}

    for att in dir(obj):
        if not att.startswith("_"):
            print(att)
            dic[att] = getattr(obj, att)
    return dic


def estimate_time(offset_time, explored_percent):
    if explored_percent == 0:
        return ""
    estimated = offset_time / explored_percent
    for t in ["s", "m", "h"]:
        if t == "h":
            break
        if estimated > 60:
            estimated /= 60
        else:
            break
    return f" estimated time: {estimated:.2f}{t}"
