import argparse

import stormpy
import stormpy.synthesis
import stormpy.pomdp
from stormpy import LongRunAvarageOperator, ExplicitQualitativeCheckResult
import re
import decimal


def print_mdp(mdp_model):
    print(mdp_model.transition_matrix)
    for state in mdp_model.states:
        if state.id in mdp_model.initial_states:
            print("State {} is initial".format(state.id))
        for action in state.actions:
            for transition in action.transitions:
                print("From state {} by action {}, with probability {}, go to state {}".format(state, action,
                                                                                               transition.value(),
                                                                                               transition.column))


class Hole:
    def __init__(self, observation, actions):
        self.actions = actions
        self.observation = observation
        self.selected_action_index = 0
        self.selected_action = actions[0]

    def __repr__(self):
        return f"observation {self.observation}: {self.actions}"

    def select_next_action(self):
        self.selected_action_index += 1
        self.selected_action = self.actions[self.selected_action_index]
        return self.selected_action

    def can_select_next_action(self):
        return self.selected_action_index + 1 < len(self.actions)


class PrismPomdp:
    def __init__(self, file_path):
        self.prism_program = stormpy.parse_prism_program(file_path)
        self.model = stormpy.build_model(self.prism_program)
        # self.model = stormpy.pomdp.make_canonic(self.model)
        # ^ this also asserts that states with the same observation have the
        # same number and the same order of available actions
        print(self.model.transition_matrix)
        self.design_space = self.create_design_space()

    def __iter__(self):
        self.nr_actions = self.model.transition_matrix.nr_rows
        self.initial_assignment = len(self.design_space) * [0]
        self.current_assignment = None
        return self

    def __next__(self):
        if self.current_assignment is None:
            self.current_assignment = self.initial_assignment
        else:
            self.update_assignment()

        return self.assignment_to_bv()

    def create_design_space(self):
        print(self.model)
        holes = []
        seen_observations = []
        observations = self.model.observations
        observation_mr = self.model.nr_observations
        for state in self.model.states:
            obs = self.model.get_observation(state.id)
            if obs not in seen_observations:
                action_indices = []
                for act in state.actions:
                    action_indices.append(act.id)
                holes.append(Hole(obs, action_indices))

            # choice_index = self.model.get_choice_index(0, 0)
            seen_observations.append(obs)
        return holes

    def assignment_to_bv(self):
        selected_actions = []
        for state in self.model.states:
            obs_at_state = self.model.get_observation(state.id)
            action_at_observation = self.get_action_at_obs(obs_at_state)
            for action in state.actions:
                if action.id == action_at_observation:
                    choice_index = self.model.get_choice_index(state.id, action_at_observation)
                    selected_actions.append(choice_index)
        return stormpy.BitVector(self.nr_actions, selected_actions)

    def update_assignment(self):
        for index, obs in enumerate(self.design_space):
            if obs.can_select_next_action():
                obs.select_next_action()
                return
        raise StopIteration

    def get_action_at_obs(self, observation_id):
        for index, hole in enumerate(self.design_space):
            if hole.observation == observation_id:
                return hole.selected_action


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    assert args.path is not None
    return args.path


class Dtmc:
    def __init__(self, pomdp, choice):
        self.pomdp = pomdp
        self.mdp = self.pomdp_as_mdp()
        self.mdp = self.restrict_mdp(choice)
        self.dtmc = self.mdp_as_dtmc()

    def restrict_mdp(self, choice):
        keep_unreachable_states = False  # TODO investigate this
        all_states = stormpy.BitVector(self.mdp.nr_states, True)
        options = stormpy.SubsystemBuilderOptions()
        options.build_action_mapping = True
        submodel_construction = stormpy.construct_submodel(
            self.mdp, all_states, choice, keep_unreachable_states, options
        )

        model = submodel_construction.model
        state_map = list(submodel_construction.new_to_old_state_mapping)
        choice_map = list(submodel_construction.new_to_old_action_mapping)
        return model

    def pomdp_as_mdp(self):
        # tm = mdp.transition_matrix
        # tm.make_row_grouping_trivial()
        pomdp = self.pomdp
        components = stormpy.storage.SparseModelComponents(pomdp.transition_matrix, pomdp.labeling, pomdp.reward_models)
        return stormpy.storage.SparseMdp(components)

    def mdp_as_dtmc(self):
        tm = self.mdp.transition_matrix
        tm.make_row_grouping_trivial()
        mdp = self.mdp
        components = stormpy.storage.SparseModelComponents(mdp.transition_matrix, mdp.labeling, mdp.reward_models)
        return stormpy.storage.SparseDtmc(components)


def analyze_model(model, prism_program, formula):
    # print_mdp(model)
    properties = stormpy.parse_properties_for_prism_program(formula, prism_program)
    result = stormpy.model_checking(model, properties[0])
    return result.at(0)


if __name__ == '__main__':
    path = get_args()
    pomdp = PrismPomdp(path)
    for choice in pomdp:
        dtmc = Dtmc(pomdp.model, choice)
        a = analyze_model(dtmc.dtmc, pomdp.prism_program, 'LRA=? [ \"goal\" ]')
