import numpy as np
from src.fingering.constraints import HandConstraints

class DPSolver:
    def __init__(self, constraints=None):
        self.constraints = constraints or HandConstraints()

    def solve(self, events):
        '''
        input: list of event objects
        output: list of fingerings
        '''
        # THIS IS MVP. REPLACE WITH VITERBI ALGORITHM
        results = []
        for e in events:
            n_notes = len(e['pitches'])
            if n_notes == 1:
                results.append([1])
            elif n_notes == 2:
                results.append([1, 5])
            elif n_notes == 3:
                results.append([1, 3, 5])
            else: 
                results.append([1, 2, 3, 5][:n_notes])
        return results