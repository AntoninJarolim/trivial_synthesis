import math
import threading
import time
import stormpy.synthesis
import stormpy.pomdp
from stormpy import BuilderOptions
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
    def __init__(self, observation, options, memory):
        self.observation = observation
        self.memory = memory
        self.selected_option_index = 0
        self.options = options
        self.options_size = len(self.options)
        self.repr_str = ""

    @property
    def selected_option(self):
        return self.options[self.selected_option_index]

    def select_next_option(self) -> bool:
        overflow = self.last_option_selected()
        self.selected_option_index = 0 if overflow else self.selected_option_index + 1
        return overflow

    def last_option_selected(self) -> bool:
        return self.selected_option_index + 1 == self.options_size

    def selected_str(self):
        return f" --> {self.options[self.selected_option_index]}"

    def __repr__(self):
        return f"{self.repr_str}({self.observation.label}, mem: {self.memory})={self.options} " + self.selected_str()

    def __str__(self, selected_index=None):
        if selected_index is None:
            return self.__repr__()
        return f"{self.repr_str}({self.observation.label}, mem: {self.memory})={self.options[selected_index]} "


class ActionHole(Hole):
    def __init__(self, observation, actions, memory: int):
        super().__init__(observation, actions, memory)
        self.repr_str = 'A'


class MemoryHole(Hole):
    def __init__(self, observation, options, memory: int):
        super().__init__(observation, options, memory)
        self.repr_str = 'M'


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
        self.prism_program = stormpy.parse_prism_program(file_path)
        self.model = self.build_model()
        self.unfolded = self.unfold_memory(memory_size)
        self.unfolded_states = self.crate_unfolded_states()
        self.choice_labeling = self.model.choice_labeling
        self.observation_valuations = self.model.observation_valuations

    def build_model(self):
        builder = BuilderOptions()
        builder.set_build_choice_labels(True)
        builder.set_build_observation_valuations(True)
        model = stormpy.build_sparse_model_with_options(self.prism_program, builder)
        return stormpy.pomdp.make_canonic(model)
        # ^ this also asserts that states with the same observation have the
        # same number and the same order of available actions

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

    def pomdp_as_mdp(self):
        # tm = mdp.transition_matrix
        # tm.make_row_grouping_trivial()
        pomdp = self.model
        components = stormpy.storage.SparseModelComponents(pomdp.transition_matrix, pomdp.labeling, pomdp.reward_models)
        return stormpy.storage.SparseMdp(components)


class DesignSpace:
    def __init__(self, model: Pomdp):
        self.pomdp = model
        self.nr_actions = self.pomdp.unfolded.transition_matrix.nr_rows
        self.current_assignment = None

        self.action_holes, seen_observations = self.create_action_holes()
        self.memory_holes = self.create_memory_holes(seen_observations)
        self.size = self.count_assignments()

    @property
    def design_space(self):
        return self.action_holes + self.memory_holes

    def __iter__(self):
        return self

    def __next__(self):
        # do not update at fist iteration
        if self.current_assignment is not None:
            self.update_assignment()
        self.current_assignment = [hole.selected_option_index for hole in self.design_space]
        return Assignment(self.assignment_to_bv(), self.current_assignment)

    def assignment_to_bv(self):
        selected_actions = []
        for state in self.pomdp.unfolded_states:
            action_at_observation, memory_at_observation = self.get_selection(state.observation, state.memory)
            select = action_at_observation.action.id * self.pomdp.memory_size + memory_at_observation
            choice_index = self.pomdp.unfolded.get_choice_index(state.state.id, select)
            selected_actions.append(choice_index)
        return stormpy.BitVector(self.nr_actions, selected_actions)

    def get_selection(self, observation_id, memory):
        action = None
        for hole in self.action_holes:
            if hole.observation.id == observation_id and hole.memory == memory:
                action = hole.selected_option

        memory_update = 0
        for hole in self.memory_holes:
            if hole.observation.id == observation_id and hole.memory == memory:
                memory_update = hole.selected_option

        return action, memory_update

    def update_assignment(self):
        for hole in self.design_space:
            overflow = hole.select_next_option()
            if not overflow:
                return
        raise StopIteration

    def explain_assignment(self, assignment) -> str:
        assignment_str = []
        for i, hole in enumerate(self.design_space):
            if len(hole.options) > 1:
                assignment_str.append(hole.__str__(assignment[i]))
        return ', '.join(assignment_str)

    def create_memory_holes(self, seen_observations):
        holes = []
        if self.pomdp.memory_size > 1:
            memory_options = [x for x in range(self.pomdp.memory_size)]
            for obs in seen_observations:
                for mem in range(self.pomdp.memory_size):
                    mem_hole = MemoryHole(obs, memory_options, mem)
                    holes.append(mem_hole)
        return holes

    def create_action_holes(self):
        holes = []
        seen_observations = []
        for state in self.pomdp.model.states:
            obs = Observation(state.id, self.pomdp.model, self.pomdp.observation_valuations)
            if obs.id not in [o.id for o in seen_observations]:
                actions = []
                for act in state.actions:
                    choice = self.pomdp.model.get_choice_index(state.id, act.id)
                    actions.append(Action(act, self.pomdp.choice_labeling, choice))

                for mem in range(self.pomdp.memory_size):
                    holes.append(ActionHole(obs, actions, mem))

                seen_observations.append(obs)
        return holes, seen_observations

    def count_assignments(self):
        choices = [hole.options_size for hole in self.design_space]
        return math.prod(choices)


class Dtmc:
    def __init__(self, mdp, choice):
        self.mdp = mdp
        self.mdp = self.restrict_mdp(choice)
        self.dtmc = self.mdp_as_dtmc()

    def verify_property(self, prop):
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
        # state_map = list(submodel_construction.new_to_old_state_mapping)
        # choice_map = list(submodel_construction.new_to_old_action_mapping)
        return model

    def mdp_as_dtmc(self):
        tm = self.mdp.transition_matrix
        tm.make_row_grouping_trivial()
        mdp = self.mdp
        components = stormpy.storage.SparseModelComponents(mdp.transition_matrix, mdp.labeling, mdp.reward_models)
        return stormpy.storage.SparseDtmc(components)


class Property:
    def __init__(self, formula: str):
        self.formula = formula
        self.property = self.parse_property(formula)
        self.exact_formula = exact_specifications(formula)
        self.exact_property = self.parse_property(self.exact_formula)
        self.optimizing = '?' in formula  # Is optimizing if contains '?'
        self.maximizing = 'max' in formula

    def __repr__(self):
        return self.formula

    @staticmethod
    def parse_property(formula):
        # properties = stormpy.parse_properties_for_prism_program(formula, prism_program)
        properties = stormpy.parse_properties(formula)
        return properties[0]


class Result:
    def __init__(self, optimizing_property: Property):
        self.maximizing_optimum = optimizing_property.maximizing if optimizing_property is not None else None
        self.satisfying_assignment = None
        self.optimal_value = None
        self.assignment = None

    def update_assignment(self, assignment, optimality_result):
        # if there is no optimization property, then do update always
        if self.maximizing_optimum is None:
            self.satisfying_assignment = assignment
        else:  # do update only if improves optimum
            if self.improves_optimum(optimality_result):
                self.optimal_value = optimality_result
                self.satisfying_assignment = assignment

    def improves_optimum(self, optimality_result):
        if self.optimal_value is None:
            return True
        if self.maximizing_optimum:
            return self.optimal_value < optimality_result
        return self.optimal_value > optimality_result


class Synthesizer:
    def __init__(self, design_space, specification):
        self.explored = 0
        self.design_space = design_space
        self.specification = specification
        self.properties = [Property(formula) for formula in specification]
        self.optimizing_property = self.find_optimizing_property()

    def verify_dtmc(self, dtmc):
        satisfying = True
        for prop in self.properties:
            satisfying &= dtmc.verify_property(prop.property)

        optimizing_value = None
        if satisfying and self.optimizing_property is not None:
            optimizing_value = dtmc.verify_property(self.optimizing_property.property)
        return satisfying, optimizing_value

    def double_check(self, model, bv):
        results = []
        dtmc = Dtmc(model, bv)
        stormpy.export_to_drn(dtmc.dtmc, "model-trivial.drn")
        for prop in self.properties:
            result = dtmc.verify_property(prop.exact_property)
            results.append(result)
        return results

    def run(self):
        result = Result(self.optimizing_property)
        for assignment in self.design_space:
            dtmc = Dtmc(self.design_space.pomdp.unfolded, assignment.bv)
            satisfying, optimality_result = self.verify_dtmc(dtmc)
            if satisfying:
                result.update_assignment(assignment, optimality_result)
                # self.print_assignment(assignment, "Found satisfying assignment: ")
            self.explored += 1
        return result

    def print_assignment(self, satisfying_assignment, message=None):
        if message is not None:
            print(message)
        print(self.design_space.explain_assignment(satisfying_assignment.assignment))

    def find_optimizing_property(self):
        optimizing_property = None
        for p in self.properties:
            if p.optimizing:
                if optimizing_property is None:
                    optimizing_property = p
                else:
                    raise Exception("More than one optimizing property found!")
        if optimizing_property is not None:
            self.properties.remove(optimizing_property)
        return optimizing_property


class TimedSynthesizer(Synthesizer):
    def __init__(self, design_space, specification):
        super().__init__(design_space, specification)
        self.start_time = None
        self.progress_scheduler = None
        self.finished = False

    def run(self):
        self.start_tracking_progress()
        result = None
        try:
            result = super().run()
        except Exception:
            print(Exception)
            self.finish()
            exit()
        self.finish()
        return result

    def finish(self):
        self.finished = True
        self.print_progress()

    def start_tracking_progress(self):
        self.start_time = time.time()
        self.schedule_process_print()

    def schedule_process_print(self):
        if not self.finished:
            self.print_progress()
            self.progress_scheduler = threading.Timer(3, self.schedule_process_print)
            self.progress_scheduler.start()

    def print_progress(self):
        offset_time = time.time() - self.start_time
        explored_part = self.explored / self.design_space.size
        print(f"{offset_time:.2f}s: explored {self.explored} out of {self.design_space.size}"
              f" -> {explored_part * 100:.6f}%" +
              estimate_time(offset_time, explored_part))


def run_synthesis():
    template_path, specification, memory_size = get_args()
    pomdp = Pomdp(template_path, memory_size)
    design_space = DesignSpace(pomdp)
    synthesizer = TimedSynthesizer(design_space, specification)

    print("\n---------------- Synthesis initiated ----------------\n")
    result = synthesizer.run()

    print("\n---------------- Synthesis completed ----------------\n")
    if result.satisfying_assignment is None:
        print("Satisfying assignment was not found.")
    else:
        double_check_result = synthesizer.double_check(design_space.pomdp.unfolded, result.satisfying_assignment.bv)
        results_str = ", ".join(str(r) for r in double_check_result)
        print(f"Double-checking: {results_str} : {result.optimal_value}")

        print("Last satisfying assignment:")
        print(design_space.explain_assignment(result.satisfying_assignment.assignment))


if __name__ == '__main__':
    run_synthesis()
