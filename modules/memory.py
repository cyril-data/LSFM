
import random

import csv

class Memory:
        
    def __init__(self, max_memory, buffer_input=None):
        self._max_memory = max_memory
        
        if buffer_input==None : self._samples = []
        else : self._samples = buffer_input
            
    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)
            
    def reset(self):
        self._samples = []
        
    def write(self, file) : 
        
        with open(file, 'w', newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["observation", "action_mask", "action", "reward","new_observation", "finished"],
            )
            writer.writeheader()
            
            for idrow,row in enumerate(self._samples) : 
                
                observation = f"[{','.join(str(float(x)) for x in  row[0])}]"
                action_mask = f"[{','.join(str(int(x)) for x in  row[1])}]"
                action = row[2]
                reward = row[3]
                new_observation = f"[{','.join(str(float(x)) for x in  row[4])}]"
                finished = row[5]
                
                writer.writerow(
                {
                    "observation": observation,
                    "action_mask": action_mask,
                    "action": action,
                    "reward": reward,
                    "new_observation": new_observation,
                    "finished": finished,
                })

    @property
    def num_samples(self):
        return len(self._samples)