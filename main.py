import stormpy
import stormpy.synthesis
import stormpy.pomdp
from stormpy import Environment, BuilderOptions

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
    def __init__(self, file_path, memory_size=1):
        self.observation_states = None
        self.prism_program = stormpy.parse_prism_program(file_path)

        self.model = self.build_model()
        self.unfolded = self.unfold_memory(memory_size)

        # atrs = get_all_attrs(self.unfolded)

        self.design_space = self.create_design_space()

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

    def unfold_memory(self, memory_size):
        if memory_size < 2:
            return

        # mark perfect observations
        # self.observation_states = [0 for obs in range(self.observations)]
        # for state in range(self.pomdp.nr_states):
        #     obs = self.pomdp.observations[state]
        #     self.observation_states[obs] += 1

        self.pomdp_manager = stormpy.synthesis.PomdpManager(self.model)

        for obs in range(len(self.model.observations)):
            # mem = self.observation_memory_size[obs]
            self.pomdp_manager.set_observation_memory_size(obs, memory_size)

        return self.pomdp_manager.construct_mdp()

    def build_model(self):
        builder = BuilderOptions()
        builder.set_build_choice_labels(True)
        model = stormpy.build_sparse_model_with_options(self.prism_program, builder)
        return stormpy.pomdp.make_canonic(model)
        # ^ this also asserts that states with the same observation have the
        # same number and the same order of available actions


class Dtmc:
    def __init__(self, pomdp, choice):
        self.pomdp = pomdp
        self.mdp = self.pomdp_as_mdp()
        self.mdp = self.restrict_mdp(choice)
        self.dtmc = self.mdp_as_dtmc()

    def restrict_mdp(self, choice):
        keep_unreachable_states = True  # TODO investigate this
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


class Synthesizer:
    def __init__(self, design_space, specification):
        self.design_space = design_space
        self.specification = specification

    def analyze_model(self, model, formula):
        # properties = stormpy.parse_properties_for_prism_program(formula, prism_program)
        properties = stormpy.parse_properties(formula)
        prop = properties[0]
        result = stormpy.model_checking(model, prop)
        return result.at(0)

    def verify_dtmc(self, dtmc, specification):
        synthesis_result = True
        for prop in specification:
            synthesis_result &= self.analyze_model(dtmc, prop)
        return synthesis_result

    def double_check(self, model, bv, specification):
        specification = exact_specifications(specification)
        results = []
        dtmc = Dtmc(model, bv)
        stormpy.export_to_drn(dtmc.dtmc, "model-trivial.drn")
        for prop in specification:
            result = self.analyze_model(dtmc.dtmc, prop)
            results.append(result)
        return results

    def run(self):
        satisfying_assignment = None
        for choice in self.design_space:
            dtmc = Dtmc(self.design_space.model, choice.bv)
            result = self.verify_dtmc(dtmc.dtmc, self.specification)
            if result is True:
                satisfying_assignment = choice
                print("Found satisfying assignment: ",
                      self.design_space.explain_assignment(satisfying_assignment.assignment))
        return satisfying_assignment


def run_synthesis():
    template_path, specification, memory_size = get_args()
    design_space = DesignSpace(template_path, memory_size)
    synthesizer = Synthesizer(design_space, specification)

    print("\n---------------- Synthesis initiated ----------------\n")
    satisfying_assignment = synthesizer.run()

    print("\n---------------- Synthesis completed ----------------\n")
    results = synthesizer.double_check(design_space.model, satisfying_assignment.bv, specification)
    results_str = ", ".join(str(r) for r in results)
    print(f"Double-checking: {results_str}")
    print("Last satisfying assignment:")
    print(design_space.explain_assignment(satisfying_assignment.assignment))


if __name__ == '__main__':
    run_synthesis()
