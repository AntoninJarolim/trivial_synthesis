import stormpy
import stormpy.synthesis
import stormpy.pomdp
from utils import *


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


class Choice:
    def __init__(self, bv, assignment):
        self.assignment = assignment
        self.bv = bv


class DesignSpace:
    def __init__(self, file_path):
        self.prism_program = stormpy.parse_prism_program(file_path)
        self.model = stormpy.build_model(self.prism_program)
        self.design_space = self.create_design_space()
        # self.model = stormpy.pomdp.make_canonic(self.model)
        # ^ this also asserts that states with the same observation have the
        # same number and the same order of available actions

    def __iter__(self):
        self.nr_actions = self.model.transition_matrix.nr_rows
        self.current_assignment = None
        return self

    def __next__(self):
        # do not update at fist iteration
        if self.current_assignment is not None:
            self.update_assignment()
        self.current_assignment = [hole.selected_action for hole in self.design_space]
        return Choice(self.assignment_to_bv(), self.current_assignment)

    def create_design_space(self):
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

    def explain_assignment(self, assignment) -> str:
        assignment_str = []
        for i, hole in enumerate(self.design_space):
            if len(hole.actions) > 1:
                assignment_str.append(f"observation {hole.observation} -> {assignment[i]}")
        return ', '.join(assignment_str)


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


def analyze_model(model, formula):
    properties = stormpy.parse_properties(formula)
    prop = properties[0]
    result = stormpy.model_checking(model, prop)
    return result.at(0)


def verify_dtmc(dtmc, specification):
    synthesis_result = True
    for prop in specification:
        synthesis_result &= analyze_model(dtmc.dtmc, prop)
    return synthesis_result


def double_check(model, bv, specification):
    specification = exact_specifications(specification)
    results = []
    dtmc = Dtmc(model, bv)
    for prop in specification:
        result = analyze_model(dtmc.dtmc, prop)
        results.append(result)
    return results


if __name__ == '__main__':
    template_path, specification = get_args()
    design_space = DesignSpace(template_path)

    print("Synthesis initiated.")
    satisfying_assignment = None
    for choice in design_space:
        dtmc = Dtmc(design_space.model, choice.bv)
        result = verify_dtmc(dtmc, specification)
        if result is True:
            satisfying_assignment = choice
    print("\n---------------- Synthesis completed ----------------\n")
    results = double_check(design_space.model, satisfying_assignment.bv, specification)
    results_str = ", ".join(str(r) for r in results)
    print(f"Double-checking: {results_str}")
    print("Satisfying assignment:")
    print(design_space.explain_assignment(satisfying_assignment.assignment))
