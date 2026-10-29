import json, tqdm
import numpy as np
from time import datetime

from src.agents import BeliefLearner, RandomAgent
from src.environment import FootballEnv


def save_results_in_json(results):
    filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

def train_agents(playerA, playerB, env, num_episodes=1_000_000):
    state = env.reset()
    for step in tqdm(range(num_episodes), desc="Training Progress", unit="episode"):
        actionA = playerA.choose_action(state)
        actionB = playerB.choose_action(state)
        next_state, rewardA, rewardB, done = env.action(actionA, actionB)
        playerA.update(state, actionA, actionB, rewardA, rewardB, next_state, done)
        playerB.update(state, actionB, actionA, rewardB, rewardA, next_state, done)
        
        ALPHA_DECAY = 0.9999954  # Decay factor for the learning rate
        playerA.alpha = playerA.alpha * ALPHA_DECAY  # Decay the learning rate
        playerB.alpha = playerB.alpha * ALPHA_DECAY  # Decay the learning rate
        if done:
            state = env.reset()  # Reset the environment for the next episode
        else:
            state = next_state

def test_agents(playerA, playerB, env, num_episodes=100_000):
    playerA.epsilon = 0.0  # Disable exploration for testing
    playerB.epsilon = 0.0  # Disable exploration for testing
    state = env.reset()
    total_rewardA = 0
    total_rewardB = 0
    num_wins_A = 0
    num_wins_B = 0
    done_count = 0
    for step in tqdm(range(num_episodes), desc="Testing Progress", unit="episode"):
        actionA = playerA.choose_action(state)
        actionB = playerB.choose_action(state)
        next_state, rewardA, rewardB, done = env.action(actionA, actionB)
        total_rewardA += rewardA
        total_rewardB += rewardB
        if rewardA > 0:
            num_wins_A += 1
        elif rewardB > 0:
            num_wins_B += 1
        if done:
            done_count += 1
            state = env.reset()  # Reset the environment for the next episode
        else:
            state = next_state
    return done_count, total_rewardA, total_rewardB, num_wins_A, num_wins_B

def main():
    message = """
    ======================================================================================
    ** Multi-Agent Reinforcement Learning in Football Environment **
    ======================================================================================
    This program simulates a football environment where two agents, Player A and Player B, 
    learn to play football using reinforcement learning techniques.
    
    List of the test:
    1) Random Learning:    Belief Learner vs Random Agent
    2) Identical Learning: Belief Learner vs Belief Learner"""
    print(message)
    np.random.seed(88)  # Set a random seed for reproducibility
    env = FootballEnv(width=5, height=4)
    print("======================================================================================")
    print("* Starting training for Belief Learner vs Random Agent... *")
    
    playerA_vs_random = BeliefLearner()
    random_agent = RandomAgent()
    train_agents(playerA_vs_random, random_agent, env, num_episodes=1_000_000)
    print("* Training completed for Belief Learner vs Random Agent. *")
    print("======================================================================================")
    print("* Starting testing for Belief Learner vs Random Agent... *")
    done_count, total_rewardA, total_rewardB, num_wins_A, num_wins_B = test_agents(playerA_vs_random, random_agent, env, num_episodes=100_000)
    perc_wins_A, perc_wins_B = (num_wins_A/done_count * 100, num_wins_B/done_count * 100) if done_count > 0 else (0, 0)
    print(f"* Testing completed for Belief Learner vs Random Agent. *")
    print(f"Match Played: {done_count}")
    print(f"Total reward: Belief Learner vs Random Agent: {total_rewardA} vs {total_rewardB}")
    print(f"Percentage of wins: Belief Learner vs Random Agent: {perc_wins_A:.2f}% vs {perc_wins_B:.2f}%")
    results = {
        "random": {
            "n": done_count,
            "rew_A": total_rewardA,
            "rew_B": total_rewardB,
            "%_A": perc_wins_A,
            "%_B": perc_wins_B,
        }
    }

    playerA = BeliefLearner()
    playerB = BeliefLearner()
    print("======================================================================================")
    print("* Starting training for Belief Learner vs Belief Learner... *")
    train_agents(playerA, playerB, env, num_episodes=1_000_000)
    print("* Training completed for Belief Learner vs Belief Learner. *")
    print("======================================================================================")
    print("* Starting testing for Belief Learner vs Belief Learner... *")
    done_count, total_rewardA, total_rewardB, num_wins_A, num_wins_B = test_agents(playerA, playerB, env, num_episodes=100_000)
    perc_wins_A, perc_wins_B = (num_wins_A/done_count * 100, num_wins_B/done_count * 100) if done_count > 0 else (0, 0)
    print(f"* Testing completed for Belief Learner vs Belief Learner. *")
    print(f"Match Played: {done_count}")
    print(f"Total reward: Belief Learner vs Belief Learner: {total_rewardA} vs {total_rewardB}")
    print(f"Percentage of wins: Belief Learner vs Belief Learner: {perc_wins_A:.2f}% vs {perc_wins_B:.2f}%")
    results["identical"] = {
        "n": done_count,
        "rew_A": total_rewardA,
        "rew_B": total_rewardB,
        "%_A": perc_wins_A,
        "%_B": perc_wins_B,
    }
    save_results_in_json(results)

if __name__ == "__main__":
    main()