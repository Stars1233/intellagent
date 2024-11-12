from typing import List
from pydantic import BaseModel, Field
from simulator.utils import set_llm_chain, batch_invoke, set_callbck
from typing import Tuple
from simulator.utils import get_llm
import networkx as nx
import random
from simulator.env import Env
from dataclasses import dataclass


class Rank(BaseModel):
    """The Rank"""
    score: int = Field(description="The final score between 0-10")


class FlowsList(BaseModel):
    """The list of flows"""
    sub_flows: List[str] = Field(description="A list of sub-flows")


class Policy(BaseModel):
    """The policy"""
    policy: str = Field(description="The policy")
    category: str = Field(description="The category of the policy")
    challenge_score: int = Field(description="The challenge score of the policy")


class PoliciesList(BaseModel):
    """The policies list"""
    policies: List[Policy] = Field(description="A list of policies and guidelines")


class EventDescription(BaseModel):
    event_description: str = Field(description="The event description")
    expected_behaviour: str = Field(description="The expected behaviour of the chatbot according to the policies")


@dataclass
class Description:
    """
    The description of the event
    """
    event_description: str
    expected_behaviour: str
    policies: List[str]
    challenge_level: int


class DescriptionGenerator:
    """
    This class is responsible for generating descriptions
    """

    def __init__(self, config: dict, environment: Env):
        """
        Initialize the descriptor generator.
        :param llm (BaseChatModel): The language model to use.
        :param config: The configuration of the class
        :param environment: The environment of the simulator
        """
        self.config = config
        self.total_cost = 0
        self.prompt = environment.prompt
        self.task_description = environment.task_description
        llm = get_llm(self.config['llm_description'])
        self.llm_description = set_llm_chain(llm, structure=EventDescription,
                                             **self.config['description_config']['prompt'])
        if self.config['refinement_config']['do_refinement']:
            llm = get_llm(self.config['llm_refinement'])
            self.feedback_chain = set_llm_chain(llm, **self.config['refinement_config']['prompt_feedback'])
            self.refinement_chain = set_llm_chain(llm, **self.config['refinement_config']['prompt_refinement'])

    def generate_policies_graph(self, override=False):
        """
        Generate the policies graph
        """
        if override or not hasattr(self, 'flows'):
            self.flows = self.extract_flows()
        if override or not hasattr(self, 'policies'):
            self.policies = self.extract_policies()
        if override or not hasattr(self, 'relations'):
            self.extract_graph()

    def extract_flows(self):
        """
        Extract the flows from the prompt
        """
        llm = get_llm(self.config['llm_policy'])
        flow_extractor = set_llm_chain(llm, structure=FlowsList, **self.config['flow_config']['prompt'])
        result = batch_invoke(flow_extractor.invoke,
                           [{'user_prompt': self.prompt}], num_workers=1,
                              callbacks=[set_callbck(self.config['llm_policy']['type'])])[0]
        self.total_cost += result['usage']
        flows = result['result']
        return flows.dict()['sub_flows']

    def extract_policies(self):
        """
        Extract the policies from the prompt
        """
        llm = get_llm(self.config['llm_policy'])
        policy_extractor = set_llm_chain(llm, **self.config['policies_config']['prompt'], structure=PoliciesList)
        flows_policies = {}
        batch = []
        for flow in self.flows:
            batch.append({'user_prompt': self.prompt, 'flow': flow})
        res = batch_invoke(policy_extractor.invoke, batch, num_workers=self.config['policies_config']['num_workers'],
                           callbacks=[set_callbck(self.config['llm_policy']['type'])])

        for i, result in enumerate(res):
            if result['error'] is not None:
                print(f"Error in sample {result['index']}: {result['error']}")
                continue
            self.total_cost += result['usage']
            flows_policies[self.flows[result['index']]] = result['result'].dict()['policies']
        return flows_policies

    def extract_graph(self):
        """
        Extract the weighted relations between the policies
        """
        llm = get_llm(self.config['llm_edge'])
        self.graph_info = {'G': nx.Graph()}

        def policy_to_str(policy):
            return f"Flow: {policy['flow']}\npolicy: {policy['policy']}"

        edge_llm = set_llm_chain(llm, structure=Rank, **self.config['edge_config']['prompt'])
        callback = set_callbck(self.config['llm_edge']['type'])
        samples_batch = []
        policies_list = []
        for flow, policies in self.policies.items():
            policies_list += [{'flow': flow, 'policy': policy['policy'], 'score': policy['challenge_score']}
                              for policy in policies]
        for i, first_policy in enumerate(policies_list):
            for j, second_policy in enumerate(policies_list[i + 1:]):
                samples_batch.append({'policy1': policy_to_str(first_policy),
                                      'policy2': policy_to_str(second_policy),
                                      'ind1': i,
                                      'ind2': j + i + 1})
        self.graph_info['nodes'] = policies_list
        num_workers = self.config['edge_config'].get('num_workers', 1)
        res = batch_invoke(edge_llm.invoke, samples_batch, num_workers=num_workers,
                           callbacks=[callback])
        all_edges = []
        for result in res:
            if result['error'] is not None:
                print(f"Error in sample {result['index']}: {result['error']}")
                continue
            self.total_cost += result['usage']
            cur_sample = samples_batch[result['index']]
            all_edges.append((cur_sample['ind1'], cur_sample['ind2'], {'weight': result['result'].score}))

        self.graph_info['G'].add_edges_from(all_edges)

    def sample_from_graph(self, threshold) -> Tuple[list, int]:
        """
        Sample a path from the graph. Traverse the graph according to edge weight probability until the path sum exceeds the threshold.
        :param threshold:
        :return: list of nodes in the path and the path sum
        """
        # Start with a random node
        current_node = random.choice(list(self.graph_info['G'].nodes))
        path = [current_node]
        path_sum = self.graph_info['nodes'][current_node]['score']

        # Traverse until the path sum exceeds the threshold
        while path_sum <= threshold:
            # Get neighbors and weights for current node
            neighbors = list(self.graph_info['G'].neighbors(current_node))
            weights = [self.graph_info['G'][current_node][neighbor]['weight'] for neighbor in neighbors]

            # Weighted choice of the next node
            next_node = random.choices(neighbors, weights=weights)[0]

            # Add the chosen node to the path and update path sum
            path.append(next_node)
            path_sum += self.graph_info['nodes'][current_node]['score']

            # Move to the next node
            current_node = next_node
        return [self.graph_info['nodes'][t] for t in path], path_sum

    def sample_description(self, challenge_complexity: int or list[int], num_samples: int = 1) -> list[Description]:
        """
        Sample a description of event
        :param challenge_complexity: The complexity of the generated description (it will be at least the provided number), either list with size num_samples or a single number
        :param num_samples: The number of samples to generate
        :return: The description of the event, the list of policies that were used to generate the description and the actual complexity of the description
        """

        def policies_list_to_str(policies):
            return "\n".join([f"Policy {i} flow: {policy['flow']}\nPolicy {i} content: {policy['policy']}\n------" for
                              i, policy in enumerate(policies)])

        if isinstance(challenge_complexity, int):
            challenge_complexity = [challenge_complexity] * num_samples
        elif len(challenge_complexity) != num_samples:
            raise ValueError(
                "The challenge complexity should be either a single number or a list of numbers with the same length as num_samples")

        samples_batch = []
        all_policies = []
        for i, cur_score in enumerate(challenge_complexity):
            policies, path_sum = self.sample_from_graph(cur_score)
            all_policies.append({'policies': policies, 'path_sum': path_sum})
            samples_batch.append({'task_description': self.task_description,
                                  'policies': policies_list_to_str(policies)})
        num_workers = self.config['description_config'].get('num_workers', 1)
        callback = set_callbck(self.config['llm_description']['type'])
        res = batch_invoke(self.llm_description.invoke, samples_batch, num_workers=num_workers, callbacks=[callback])
        for result in res:
            if result['error'] is not None:
                print(f"Error in sample {result['index']}: {result['error']}")
                continue
            self.total_cost += result['usage']
            all_policies[result['index']]['description'] = result['result'].event_description
            all_policies[result['index']]['expected_behaviour'] = result['result'].expected_behaviour
        descriptions = [Description(event_description=policy['description'],
                                    expected_behaviour=policy['expected_behaviour'],
                                    policies= policy['policies'],
                                    challenge_level=policy['path_sum']) for policy in all_policies]
        if self.config['refinement_config']['do_refinement']:
            descriptions = self.expected_behaviour_refinement(descriptions)
        return descriptions

    def expected_behaviour_refinement(self, descriptions: list[Description], num_iterations=1) -> list[Description]:
        """
        Verify the expected behaviour of the chatbot according to each policy
        :param descriptions:
        :param num_iterations:
        :return: new updated descriptions list
        """
        iteration_indices = list(range(len(descriptions)))
        num_workers = self.config['refinement_config'].get('num_workers', 5)
        callback = set_callbck(self.config['llm_refinement']['type'])

        for i in range(num_iterations):
            batch_input = []
            # Step 1: provide feedback
            for ind in iteration_indices:
                batch_input.append({'description': descriptions[ind].event_description,
                                    'behaviour': descriptions[ind].expected_behaviour,
                                    'prompt': self.prompt})
            res = batch_invoke(self.feedback_chain.invoke, batch_input, num_workers=num_workers, callbacks=[callback])
            cur_refine_indices = []
            improved_batch = []
            # refine the behaviour
            for j, result in enumerate(res):
                if result['error'] is not None or 'None' in result['result'].content:
                    continue
                else:
                    cur_refine_indices.append(iteration_indices[result['index']])
                    cur_batch = batch_input[result['index']]
                    cur_batch['feedback'] = result['result'].content
                    improved_batch.append(cur_batch)
                    self.total_cost += result['usage']
            res = batch_invoke(self.refinement_chain.invoke, improved_batch,
                               num_workers=num_workers, callbacks=[callback])
            for j, result in enumerate(res):
                if result['error'] is not None or 'None' in result['result'].content:
                    continue
                else:
                    descriptions[cur_refine_indices[result['index']]].expected_behaviour = result['result'].content
                    self.total_cost += result['usage']
            iteration_indices = cur_refine_indices
        return descriptions

    def __getstate__(self):
        # Return a dictionary of picklable attributes
        state = self.__dict__.copy()
        # Remove the non-picklable attribute
        del state['llm_description']
        if 'feedback_chain' in state:
            del state['feedback_chain']
        if 'refinement_chain' in state:
            del state['refinement_chain']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        llm = get_llm(self.config['llm_description'])
        self.llm_description = set_llm_chain(llm, structure=EventDescription,
                                             **self.config['description_config']['prompt'])
        if self.config['refinement_config']['do_refinement']:
            llm = get_llm(self.config['llm_refinement'])
            self.feedback_chain = set_llm_chain(llm, **self.config['refinement_config']['prompt_feedback'])
            self.refinement_chain = set_llm_chain(llm, **self.config['refinement_config']['prompt_refinement'])
        return self
