import numpy as np

class FootballEnv:
    def __init__(self, width = 5, height = 4):
        self.width = width
        self.height = height
        # Initialize the state of the environment
        self.reset()

    def action(self, actionA, actionB):
        action_dict = {
            "N": (-1, 0),
            "S": (1, 0),
            "E": (0, 1),
            "W": (0, -1),
            "0": (0, 0)
        }
        pos_A, pos_B, _ = self.state
        
        if actionA in action_dict and actionB in action_dict:
            new_pos_A = (pos_A[0] + action_dict[actionA][0], pos_A[1] + action_dict[actionA][1])
            new_pos_B = (pos_B[0] + action_dict[actionB][0], pos_B[1] + action_dict[actionB][1])
            new_pos = (new_pos_A, new_pos_B)
        else:
            raise ValueError("Invalid action. Please choose from 'N', 'S', 'E', 'W', or '0'.")

        # Randomly choose which player gets to move first (0 for A, 1 for B)
        coin_flip = np.random.choice([0, 1])
        
        # move the player with the ball first  
        moving_order = [coin_flip, 1 - coin_flip]
        for moving_player in moving_order:
            reward_0, reward_1, done = self.__move_(moving_player, new_pos)
            if done:
                break  # Stop moving if a goal is scored
        return tuple(self.state), reward_0, reward_1, done  # Return the state and the player who scored

    def reset(self):
        pos_A = (1, 3)
        pos_B = (2, 1)
        posses = np.random.choice([0, 1])  # Randomly assign the ball to one of the players
        self.state = [pos_A, pos_B, posses]
        return self.state

    def __move_(self, player, new_pos):
        # Randomly end the game with a 10% chance
        if np.random.rand() < 0.1:
            return tuple(self.state), 0, 0, True  # state, reward_A, reward_B, done
        in_goal = False
        _, _, posses = self.state
        # if the new moving player's position is the same as the other player...
        if new_pos[player] == self.state[1- player]:
            # ...then the players stays in the same position...
            self.state[2] = 1 - player # ...and the other player gets the ball
        else:
            in_bounds = new_pos[player][0] >= 0 and new_pos[player][0] < self.height \
                            and new_pos[player][1] >= 0 and new_pos[player][1] < self.width
            in_goal = (posses == player) and \
                      ((posses == 0 and new_pos[player][1] == -1) or \
                       (posses == 1 and new_pos[player][1] == self.width)) and \
                      (new_pos[player][0] == self.height // 2 - 1 or new_pos[player][0] == self.height // 2)
            if in_bounds:
                # if the new position respect the bounds, the player change position...
                self.state[player] = new_pos[player]    
        reward = [0, 0]
        if in_goal:
            reward[player] = 1
            reward[1 - player] = -1
        return reward[0], reward[1], in_goal