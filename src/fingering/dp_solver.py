import numpy as np
from typing import List, Dict, Any
from src.fingering.constraints import BiomechanicalHand, FingerState

class DPSolver:
    def __init__(self):
        self.hand = BiomechanicalHand()

    def solve(self, events: List[Dict[str, Any]]) -> List[List[int]]:
        '''
        Viterbi Algorithm for polyphonic piano fingerings
        args: list of event dicts (pitches and onset_sec)
        return: list of lists, where each inner list is fingering for that event
        '''
        if not events:
            return []
        
        n_events = len(events)

        # pre-compute states: layers[t] = list of valid FingerState objects for event t
        layers: List[List[FingerState]] = []

        print(f".  - Generating state space for {n_events} events...")
        for e in events:
            pitches = e['pitches']
            states = self.hand.get_valid_configurations(pitches)
            if not states:
                # fallback for impossible chords: use a dummy to prevent crash
                # real robot will drop notes
                dummy = tuple(range(1, len(pitches) + 1))
                states = [FingerState(tuple(pitches), dummy)]
            layers.append(states)

        # Viterbi init
        # cost_matrix[t][state_idx] = min_cost to reach node
        # parent_matrix[t][state_odx] = indx of parent in layer t-1

        cost_layers = []
        parent_layers = []

        # init first layer
        first_costs = []
        for s in layers[0]:
            # init cost is just static cost of grip
            first_costs.append(self.hand.static_cost(s))
        cost_layers.append(first_costs)
        parent_layers.append([-1] * len(layers[0]))

        # forward pass
        for t in range(1, n_events):
            prev_layer = layers[t - 1]
            curr_layer = layers[t]
            prev_costs = cost_layers[t-1]
            
            curr_costs = []
            curr_parents = []

            dt = events[t]['onset_sec'] - events[t-1]['onset_sec']

            for i, curr_state in enumerate(curr_layer):
                min_c = float('inf')
                best_p = -1

                # node static cost
                node_cost = self.hand.static_cost(curr_state)

                for j, prev_state in enumerate(prev_layer):
                    # edge cost
                    trans_cost = self.hand.transition_cost(prev_state, curr_state, dt)
                    total = prev_costs[j] + trans_cost + node_cost

                    if total < min_c:
                        min_c = total
                        best_p = j

                curr_costs.append(min_c)
                curr_parents.append(best_p)

            cost_layers.append(curr_costs)
            parent_layers.append(curr_parents)

        # backward pass (path reconstruction)
        best_path_indices = [0] * n_events

        # find best end node
        last_layer_costs = cost_layers[-1]
        best_last_idx = int(np.argmin(last_layer_costs))
        best_path_indices[-1] = best_last_idx

        # backtrack
        for t in range(n_events - 1, 0, -1):
            parent_idx = parent_layers[t][best_path_indices[t]]
            best_path_indices[t-1] = parent_idx

        # format output
        results = []
        for t, idx in enumerate(best_path_indices):
            state = layers[t][idx]
            results.append(list(state.fingers))

        return results