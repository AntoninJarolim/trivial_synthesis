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
    def __init__(self):
        self.prism_program = None
        self.model = None

    def bruh(self, file_path):
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


def build_model():
    builder = stormpy.SparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False,
                                          has_custom_row_grouping=False)
    builder.add_next_value(row=0, column=1, value=1)
    builder.add_next_value(row=1, column=0, value=1)
    transition_matrix = builder.build()

    state_labeling = stormpy.storage.StateLabeling(2)
    state_labeling.add_label('init')
    state_labeling.add_label_to_state('init', 0)
    print(state_labeling.get_states('init'))
    components = stormpy.SparseModelComponents(transition_matrix=transition_matrix, state_labeling=state_labeling)
    dtmc = stormpy.storage.SparseDtmc(components)

    print(dtmc)
    return dtmc


if __name__ == '__main__':
    path = get_args()
    # model = PrismModel(path)
    model = PrismModel()
    model.model = build_model()
    print_mdp(model.model)
    stormpy.construct_submodel()


