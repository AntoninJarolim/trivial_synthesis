import argparse

import stormpy
from stormpy import LongRunAvarageOperator, ExplicitQualitativeCheckResult
import re
import decimal


def print_mdp(mdp_model):
    for state in mdp_model.states:
        if state.id in mdp_model.initial_states:
            print("State {} is initial".format(state.id))
        for action in state.actions:
            for transition in action.transitions:
                print("From state {} by action {}, with probability {}, go to state {}".format(state, action,
                                                                                               transition.value(),
                                                                                               transition.column))


class PrismModel:
    def __init__(self, file_path):
        self.prism_program = stormpy.parse_prism_program(file_path)
        self.model = stormpy.build_model(self.prism_program)



def analyze_model(model, formula):
    properties = stormpy.parse_properties(formula)
    result = stormpy.model_checking(model.model, properties[0])
    return result.at(0)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    assert args.path is not None
    return args.path


if __name__ == '__main__':
    path = get_args()
    model = PrismModel(path)

    print_mdp(model.model)


