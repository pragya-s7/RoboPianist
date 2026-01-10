import numpy as np
from typing import List, Dict, Any, Tuple
from src.fingering.constraints import BiomechanicalHand, FingerState

class DPSolver:
    def __init__(self):
        self.hand = BiomechanicalHand()

    def _prune_chord(self, pitches: List[int]) -> List[int]:
        """
        If a chord has more notes than fingers (5), keep the most musically 
        significant ones: the Bass (lowest) and Melody (highest).
        """
        if len(pitches) <= 5:
            return pitches
        
        # Sort ascending
        sorted_p = sorted(pitches)
        
        # Heuristic: Keep lowest (bass) and top 3 (melody/structure)
        # e.g., if 6 notes: keep 0, and 3,4,5. Drop 1,2.
        # This is a standard pianist reduction technique.
        n_drop = len(pitches) - 5
        
        # We always keep index 0 (root/bass) and index -1 (melody)
        # We greedily drop from the lower-middle
        reduced = [sorted_p[0]] + sorted_p[1+n_drop:]
        
        return reduced

    def solve(self, events: List[Dict[str, Any]]) -> List[List[int]]:
        '''
        Viterbi Algorithm for polyphonic piano fingerings.
        Robust against impossible chords and impossible transitions.
        
        args: list of event dicts (pitches and onset_sec)
        return: list of lists, where each inner list is fingering for that event
                (e.g. [1, 3, 5] or [None] if rest)
        '''
        if not events:
            return []
        
        n_events = len(events)

        # --- 1. State Space Generation ---
        layers: List[List[FingerState]] = []
        
        # To handle disconnected graphs (impossible transitions), we track
        # if a layer is fully unreachable.
        
        for t, e in enumerate(events):
            raw_pitches = e.get('pitches', [])
            
            # Prune impossible chords
            playable_pitches = self._prune_chord(raw_pitches)
            
            states = self.hand.get_valid_configurations(playable_pitches)
            
            # Fallback: If strict biomechanics returns nothing (e.g. impossible stretch),
            # force a naive 1-2-3 assignment just to keep the pipeline alive.
            if not states and playable_pitches:
                # Naive assignment: 1..N
                dummy_fingers = tuple(range(1, len(playable_pitches) + 1))
                states = [FingerState(tuple(playable_pitches), dummy_fingers)]
            elif not playable_pitches:
                # Rest / Empty event
                states = [FingerState(tuple(), tuple())]
                
            layers.append(states)

        # --- 2. Viterbi Initialization ---
        # cost_matrix[t][state_idx]
        # parent_matrix[t][state_idx]
        
        cost_layers = []
        parent_layers = []

        # Init first layer
        first_costs = []
        for s in layers[0]:
            first_costs.append(self.hand.static_cost(s))
        
        cost_layers.append(first_costs)
        parent_layers.append([-1] * len(layers[0]))

        # --- 3. Forward Pass ---
        for t in range(1, n_events):
            prev_layer = layers[t - 1]
            curr_layer = layers[t]
            prev_costs = cost_layers[t-1]
            
            curr_costs = []
            curr_parents = []

            dt = events[t]['onset_sec'] - events[t-1]['onset_sec']
            
            # If previous layer was dead (all infinite costs), reset accumulation
            # This effectively segments the song, solving independent phrases.
            if all(c == float('inf') for c in prev_costs):
                prev_costs = [0.0] * len(prev_layer)

            for i, curr_state in enumerate(curr_layer):
                min_c = float('inf')
                best_p = -1

                node_cost = self.hand.static_cost(curr_state)

                for j, prev_state in enumerate(prev_layer):
                    if prev_costs[j] == float('inf'):
                        continue

                    trans_cost = self.hand.transition_cost(prev_state, curr_state, dt)
                    total = prev_costs[j] + trans_cost + node_cost

                    if total < min_c:
                        min_c = total
                        best_p = j

                curr_costs.append(min_c)
                curr_parents.append(best_p)
            
            # Safety: If current layer is unreachable from prev, pick locally best node
            # and treat as restart point.
            if all(c == float('inf') for c in curr_costs) and curr_layer:
                # Find min static cost in current layer
                best_local_idx = -1
                best_local_cost = float('inf')
                for i, s in enumerate(curr_layer):
                    c = self.hand.static_cost(s)
                    if c < best_local_cost:
                        best_local_cost = c
                        best_local_idx = i
                
                # Force entry
                if best_local_idx != -1:
                    curr_costs[best_local_idx] = best_local_cost
                    # Parent is -1 indicates a break in chain
                    curr_parents[best_local_idx] = -1

            cost_layers.append(curr_costs)
            parent_layers.append(curr_parents)

        # --- 4. Backward Pass ---
        best_path_indices = [0] * n_events

        # Find best end node
        last_layer_costs = cost_layers[-1]
        best_last_idx = int(np.argmin(last_layer_costs)) if last_layer_costs else 0
        best_path_indices[-1] = best_last_idx

        # Backtrack
        for t in range(n_events - 1, 0, -1):
            curr_idx = best_path_indices[t]
            parent_idx = parent_layers[t][curr_idx]
            
            if parent_idx == -1:
                # Chain break detected. Minimize cost of previous layer independently.
                prev_layer_costs = cost_layers[t-1]
                parent_idx = int(np.argmin(prev_layer_costs)) if prev_layer_costs else 0
            
            best_path_indices[t-1] = parent_idx

        # --- 5. Format Output ---
        results = []
        for t, idx in enumerate(best_path_indices):
            if not layers[t]:
                results.append([])
                continue
                
            state = layers[t][idx]
            
            # We need to map the solved fingers back to the ORIGINAL event pitches.
            # If we pruned the chord, we must insert None for dropped notes 
            # or ensure downstream handles mismatch. 
            # CHORD Spec: "fingering" list must match length of "pitches".
            
            original_pitches = events[t].get('pitches', [])
            solved_pitches = state.pitches
            solved_fingers = state.fingers
            
            # Map: Pitch -> Finger
            p_to_f = dict(zip(solved_pitches, solved_fingers))
            
            final_fingering = []
            for p in original_pitches:
                final_fingering.append(p_to_f.get(p, None)) # None for pruned notes
                
            results.append(final_fingering)

        return results