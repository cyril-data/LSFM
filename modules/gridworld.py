import numpy as np
import utils

class SimpleGrid(object):
    def __init__(self, size, block_pattern="empty", block_user="empty",
                 verbose=False, obs_mode="onehot"):
        self.verbose = verbose
        self.grid_size = size
        self.action_size = 4
        self.obs_mode = obs_mode
        self.state_size = size*size
        self.blocks = self.make_blocks(block_pattern)
        self.goal_pos = []
        self.reward_pos = []

        self.reward_min = 0.
        self.reward_max = 1.
        self.agent_pos = []
        self.obs_size = None
        self.done = None
        self.observations = None
        if obs_mode == "onehot":
            self.obs_size = self.state_size
            self.goal_size = self.state_size
        if obs_mode == "twohot":
            self.obs_size = self.grid_size * 2
            self.goal_size = self.grid_size * 2
        if obs_mode == "geometric":
            self.obs_size = 2
            self.goal_size = 2
        if obs_mode == "index":
            self.obs_size = 1
            self.goal_size = 1
            
    def reset(self, goal_pos=None, agent_pos=None, reward_pos=None, reward_minmax=None, reward_user_defined=False):
        self.done = False
        if goal_pos != None   : self.goal_pos = goal_pos
        else:                   self.goal_pos = [self.get_free_spot()]

        if agent_pos != None  : self.agent_pos = agent_pos
        else:                   self.agent_pos = self.get_free_spot()

        if reward_pos == None : self.reward_pos = self.goal_pos
        else : 
            if reward_user_defined : self.reward_pos = reward_pos   
            else : self.reward_pos = self.goal_pos
            


        if reward_minmax == None : 
            self.reward_min = 0.
            self.reward_max = 1.
        else : 
            self.reward_min = reward_minmax[0]
            self.reward_max = reward_minmax[1]
    

    def get_free_spot(self):
        free = False
        possible_x = np.arange(0, self.grid_size)
        possible_y = np.arange(0, self.grid_size)
        while not free:
            try_x = np.random.choice(possible_x, replace=False)
            try_y = np.random.choice(possible_y, replace=False)
            try_position = [try_x, try_y]

            if try_position not in self.all_positions:
                return try_position
        
    def make_blocks(self, patterns, user_defined=None):
        if isinstance(patterns, str) :   pattern = patterns
        else : 
            pattern = patterns[0]
            user_defined = patterns[1]
        if pattern == "four_rooms":
            mid = int(self.grid_size // 2)
            earl_mid = int(mid // 2)
            late_mid = mid+earl_mid + 1
            blocks_a = [[mid,i] for i in range(self.grid_size)]
            blocks_b = [[i,mid] for i in range(self.grid_size)]
            blocks = blocks_a + blocks_b
            self.bottlenecks = [[mid,earl_mid],[mid,late_mid],[earl_mid,mid],[late_mid,mid]]
            for bottleneck in self.bottlenecks:
                blocks.remove(bottleneck)
            return blocks
        if pattern == "user_defined":
            self.bottlenecks = []
            blocks = user_defined
            return blocks
        if pattern == "empty":
            self.bottlenecks = []
            return []
        if pattern == "random":
            blocks = []
            for i in range(self.state_size // 10):
                blocks.append([np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)])
            self.bottlenecks = []
            return blocks
        if pattern == "two_rooms":
            mid = int(self.grid_size // 2)
            blocks = [[mid,i] for i in range(self.grid_size)]
            blocks.remove([mid,mid])
            self.bottlenecks = [[mid,mid]]
            return blocks
        
    @property
    def grid(self):
        grid = np.zeros([self.grid_size, self.grid_size, 3])

        if  (self.reward_max ==  self.reward_min) and  (self.reward_min == 0) :           
            grid[:,:, 0] = 0.5     
            grid[:,:, 1] = 0.5     
            grid[:,:, 2] = 0.5
        else : 
            for reward in self.reward_pos:
                grid[reward[0], reward[1], 0] = 0.5
                grid[reward[0], reward[1], 1] = 0.5
                grid[reward[0], reward[1], 2] = 0.5

        for block in self.blocks:
            grid[block[0], block[1], 2] = 1
            grid[block[0], block[1], 0] = 0
            grid[block[0], block[1], 1] = 0


        for goal in self.goal_pos : 
            grid[goal[0], goal[1], 1] = 1
            grid[goal[0], goal[1], 0] = 0
            grid[goal[0], goal[1], 2] = 0

        grid[self.agent_pos[0], self.agent_pos[1], 0] = 1
        grid[self.agent_pos[0], self.agent_pos[1], 1] = 0
        grid[self.agent_pos[0], self.agent_pos[1], 2] = 0


        return grid
    

    def grid_state(self, state):
        grid = np.zeros([self.grid_size, self.grid_size, 3])

        if  (self.reward_max ==  self.reward_min) and  (self.reward_min == 0) :           
            grid[:,:, 0] = 0.5     
            grid[:,:, 1] = 0.5     
            grid[:,:, 2] = 0.5
        else : 
            for reward in self.reward_pos:
                grid[reward[0], reward[1], 0] = 0.5
                grid[reward[0], reward[1], 1] = 0.5
                grid[reward[0], reward[1], 2] = 0.5

        for block in self.blocks:
            grid[block[0], block[1], 2] = 1
            grid[block[0], block[1], 0] = 0
            grid[block[0], block[1], 1] = 0


        for goal in self.goal_pos : 
            grid[goal[0], goal[1], 1] = 1
            grid[goal[0], goal[1], 0] = 0
            grid[goal[0], goal[1], 2] = 0

        grid[state[0], state[1], 0] = 1
        grid[state[0], state[1], 1] = 0
        grid[state[0], state[1], 2] = 0


        return grid



    def move_agent(self, direction):
        new_pos = self.agent_pos + direction
        if self.check_target(new_pos):
            self.agent_pos = list(new_pos)
            
    def simulate(self, action):
        agent_old_pos = self.agent_pos
        reward = self.step(action)
        state = self.state
        self.agent_pos = agent_old_pos
        return state
        
    def check_target(self, target):
        x_check = target[0] > -1 and target[0] < self.grid_size
        y_check = target[1] > -1 and target[1] < self.grid_size
        block_check = list(target) not in self.blocks
        if x_check and y_check and block_check:
            return True
        else:
            return False
        
    @property
    def observation(self):
        if self.obs_mode == "onehot":
            return utils.onehot(self.agent_pos[0] * self.grid_size + self.agent_pos[1], self.state_size)
        if self.obs_mode == "twohot":
            return self.twohot(self.agent_pos, self.grid_size)
        if self.obs_mode == "geometric":
            return (2 * np.array(self.agent_pos) / (self.grid_size-1)) - 1 
        if self.obs_mode == "visual":
            return env.grid
        if self.obs_mode == "index":
            return self.agent_pos[0] * self.grid_size + self.agent_pos[1]

    @property
    def goal(self):
        # if self.obs_mode == "onehot":
        #     return utils.onehot(self.goal_pos[0] * self.grid_size + self.goal_pos[1], self.state_size)
        # if self.obs_mode == "twohot":
        #     return self.twohot(self.goal_pos, self.grid_size)
        # if self.obs_mode == "geometric":
        #     return (2 * np.array(self.goal_pos) / (self.grid_size-1)) - 1 
        # if self.obs_mode == "visual":
        #     return env.grid
        if self.obs_mode == "index":
            returns = []
            for goal in self.goal_pos : 
                returns.append(goal[0] * self.grid_size + goal[1])
            return returns
        
    @property
    def all_positions(self):
        all_positions = self.blocks + self.goal_pos + [self.agent_pos]+ self.reward_pos
        return all_positions
    
    def state_to_grid(self, state):
        vec_state = np.zeros([self.state_size])
        vec_state[state] = 1
        vec_state = np.reshape(vec_state, [self.grid_size, self.grid_size])
        return vec_state
    
    def state_to_goal(self, state):
        return utils.onehot(state, self.state_size)
    
    def state_to_point(self, state):
        a = self.state_to_grid(state)
        b = np.where(a==1)
        c = [b[0][0],b[1][0]]
        return c
    
    def state_to_obs(self, state):
        if self.obs_mode == "onehot":
            point = self.state_to_point(state)
            return utils.onehot(point[0] * self.grid_size + point[1], self.state_size)
        if self.obs_mode == "twohot":
            point = self.state_to_point(state)
            return self.twohot(point, self.grid_size)
        if self.obs_mode == "geometric":
            point = self.state_to_point(state)
            return (2 * np.array(point) / (self.grid_size-1)) - 1 
        if self.obs_mode == "visual":
            return self.state_to_grid(state)
        if self.obs_mode == "index":
            return state
            
    def step(self, action):
        # 0 - Up
        # 1 - Down
        # 2 - Left
        # 3 - Right
        move_array = np.array([0,0])
        if action == 2:
            move_array = np.array([0,-1])
        if action == 3:
            move_array = np.array([0,1])
        if action == 0:
            move_array = np.array([-1,0])
        if action == 1:
            move_array = np.array([1,0])
        self.move_agent(move_array)


        if self.agent_pos in self.goal_pos:
            self.done = True
            return 1.

        if self.agent_pos in self.reward_pos:
            return self.reward_max
        else:
            return self.reward_min


    def state_to_goal(self, state):
        return self.state_to_obs(state)
