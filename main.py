import stormpy
import stormpy.synthesis
import stormpy.pomdp
from stormpy import Environment, BuilderOptions

from utils import *


class Observation:
    def __init__(self, state_id, model, observation_valuations):
        self.id = model.get_observation(state_id)
        self.label = str(observation_valuations.get_string(self.id)).replace("\t& ", ", ")

    def __repr__(self):
        return f"id:{self.id} -> {self.label}"


class Action:
    def __init__(self, action, choice_labeling, choice):
        self.action = action

        set_of_labels = choice_labeling.get_labels_of_choice(choice)
        self.label = str(set_of_labels) if len(set_of_labels) > 0 else "no_act_label"

    def __repr__(self):
        return self.label

    def __str__(self):
        return self.__repr__()


class Hole:
    def __init__(self, observation, options):
        self.observation = observation
        self.selected_option_index = 0
        self.options = options

    def select_next_option(self) -> bool:
        overflow = self.last_option_selected()
        self.selected_option_index = 0 if overflow else self.selected_option_index + 1
        return overflow

    def last_option_selected(self) -> bool:
        return self.selected_option_index + 1 == len(self.options)

    def selected_str(self):
        return f" --> {self.options[self.selected_option_index]}"

    def get_selected_option(self):
        return self.options[self.selected_option_index]


class ActionHole(Hole):
    def __init__(self, observation, actions, memory: int, action_labels=None):
        super().__init__(observation, actions)
        self.memory = memory
        self.action_labels = action_labels

    def __repr__(self):
        return f"A({self.observation.label}, mem: {self.memory})={self.options} " + self.selected_str()


class MemoryHole(Hole):
    def __init__(self, observation, options, memory: int):
        super().__init__(observation, options)
        self.memory = memory

    def __repr__(self):
        return f"M({self.observation.label}, mem:{self.memory})={self.options}" + self.selected_str()


class Assignment:
    def __init__(self, bv, assignment):
        self.assignment = assignment
        self.bv = bv


class State:
    def __init__(self, state, observation, memory):
        self.memory = memory
        self.observation = observation
        self.state = state



class Pomdp:
    def __init__(self, file_path, memory_size=1):
        self.memory_size = memory_size
        self.observation_states = None
        self.prism_program = stormpy.parse_prism_program(file_path)

        self.model = self.build_model()
        self.unfolded = self.unfold_memory(memory_size)
        self.unfolded_states = self.crate_unfolded_states()
        self.choice_labeling = self.model.choice_labeling
        self.observation_valuations = self.model.observation_valuations

        self.design_space = self.create_design_space()

    def __iter__(self):
        self.nr_actions = self.unfolded.transition_matrix.nr_rows
        self.current_assignment = None
        return self

    def __next__(self):
        # do not update at fist iteration
        if self.current_assignment is not None:
            self.update_assignment()
        self.current_assignment = [hole.selected_option_index for hole in self.design_space]
        return Assignment(self.assignment_to_bv(), self.current_assignment)

    def create_design_space(self):
        holes, seen_observations = self.create_action_holes()
        memory_holes = self.create_memory_holes(seen_observations)
        design_space = holes + memory_holes
        return design_space

    def assignment_to_bv(self):
        selected_actions = []
        for state in self.unfolded_states:
            action_at_observation = self.action_at_observation(state.observation, state.memory)
            memory_at_observation = self.memory_at_observation(state.observation, state.memory)
            select = action_at_observation.action.id * self.memory_size + memory_at_observation
            choice_index = self.unfolded.get_choice_index(state.state.id, select)
            selected_actions.append(choice_index)
        return stormpy.BitVector(self.nr_actions, selected_actions)

    def crate_unfolded_states(self):
        states = []
        for state in self.model.states:
            obs_at_state = self.model.get_observation(state.id)
            for m in range(self.memory_size):
                unfolded_index = state.id * self.memory_size + m
                unfolded_state = self.unfolded.states[unfolded_index]
                s = State(unfolded_state, obs_at_state, m)
                states.append(s)
        return states

    def update_assignment(self):
        for index, hole in enumerate(self.design_space):
            overflow = hole.select_next_option()
            if not overflow:
                return
        raise StopIteration

    def memory_at_observation(self, observation_id, memory):
        for index, hole in enumerate(self.design_space):
            if type(hole) is MemoryHole:
                if hole.observation.id == observation_id and hole.memory == memory:
                    return hole.get_selected_option()

    def action_at_observation(self, observation_id, memory):
        for index, hole in enumerate(self.design_space):
            if type(hole) is ActionHole:
                if hole.observation.id == observation_id and hole.memory == memory:
                    return hole.get_selected_option()

    def explain_assignment(self, assignment) -> str:
        assignment_str = []
        for i, hole in enumerate(self.design_space):
            if len(hole.options) > 1:
                assignment_str.append(f"observation {hole.observation} -> {assignment[i]}")
        return ', '.join(assignment_str)

    def unfold_memory(self, memory_size):
        # No need to unfold memory if memory=1
        if memory_size < 2:
            return self.pomdp_as_mdp()

        # mark perfect observations
        # self.observation_states = [0 for obs in range(self.observations)]
        # for state in range(self.pomdp.nr_states):
        #     obs = self.pomdp.observations[state]
        #     self.observation_states[obs] += 1

        # Create pomdp manager and unfold memory to pomdp creating mdp
        pomdp_manager = stormpy.synthesis.PomdpManager(self.model)
        for obs in range(len(self.model.observations)):
            # mem = self.observation_memory_size[obs]
            pomdp_manager.set_observation_memory_size(obs, memory_size)

        return pomdp_manager.construct_mdp()

    def build_model(self):
        builder = BuilderOptions()
        builder.set_build_choice_labels(True)
        builder.set_build_observation_valuations(True)
        model = stormpy.build_sparse_model_with_options(self.prism_program, builder)
        return stormpy.pomdp.make_canonic(model)
        # ^ this also asserts that states with the same observation have the
        # same number and the same order of available actions

    def pomdp_as_mdp(self):
        # tm = mdp.transition_matrix
        # tm.make_row_grouping_trivial()
        pomdp = self.model
        components = stormpy.storage.SparseModelComponents(pomdp.transition_matrix, pomdp.labeling, pomdp.reward_models)
        return stormpy.storage.SparseMdp(components)

    def create_memory_holes(self, seen_observations):
        holes = []
        if self.memory_size > 1:
            memory_options = [x for x in range(self.memory_size)]
            for obs in seen_observations:
                for mem in range(self.memory_size):
                    mem_hole = MemoryHole(obs, memory_options, mem)
                    holes.append(mem_hole)
        return holes

    def create_action_holes(self):
        holes = []
        seen_observations = []
        for state in self.model.states:
            obs = Observation(state.id, self.model, self.observation_valuations)
            if obs.id not in [o.id for o in seen_observations]:
                actions = []
                for act in state.actions:
                    choice = self.model.get_choice_index(state.id, act.id)
                    actions.append(Action(act, self.choice_labeling, choice))

                for mem in range(self.memory_size):
                    holes.append(ActionHole(obs, actions, mem))

                seen_observations.append(obs)
        return holes, seen_observations


class Dtmc:
    def __init__(self, mdp, choice):
        self.mdp = mdp
        self.mdp = self.restrict_mdp(choice)
        self.dtmc = self.mdp_as_dtmc()

    def verify_property(self, formula):
        # properties = stormpy.parse_properties_for_prism_program(formula, prism_program)
        properties = stormpy.parse_properties(formula)
        prop = properties[0]
        result = stormpy.model_checking(self.dtmc, prop)
        return result.at(0)

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

    def verify_dtmc(self, dtmc):
        synthesis_result = True
        for prop in self.specification:
            synthesis_result &= dtmc.verify_property(prop)
        return synthesis_result

    def double_check(self, model, bv):
        specification = exact_specifications(self.specification)
        results = []
        dtmc = Dtmc(model, bv)
        stormpy.export_to_drn(dtmc.dtmc, "model-trivial.drn")
        for prop in specification:
            result = dtmc.verify_property(prop)
            results.append(result)
        return results

    def run(self):
        satisfying_assignment = None
        for assignment in self.design_space:
            dtmc = Dtmc(self.design_space.unfolded, assignment.bv)
            result = self.verify_dtmc(dtmc)
            if result is True:
                satisfying_assignment = assignment
                print("Found satisfying assignment: ",
                      self.design_space.explain_assignment(satisfying_assignment.assignment))
        return satisfying_assignment


def run_synthesis():
    template_path, specification, memory_size = get_args()
    design_space = Pomdp(template_path, memory_size)
    synthesizer = Synthesizer(design_space, specification)

    print("\n---------------- Synthesis initiated ----------------\n")
    satisfying_assignment = synthesizer.run()

    print("\n---------------- Synthesis completed ----------------\n")
    if satisfying_assignment is None:
        print("Satisfying assignment was not found.")
    else:
        results = synthesizer.double_check(design_space.unfolded, satisfying_assignment.bv)
        results_str = ", ".join(str(r) for r in results)
        print(f"Double-checking: {results_str}")

        print("Last satisfying assignment:")
        print(design_space.explain_assignment(satisfying_assignment.assignment))


if __name__ == '__main__':
    run_synthesis()
