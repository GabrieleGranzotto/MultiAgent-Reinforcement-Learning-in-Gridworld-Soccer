import json
from tqdm import tqdm
import numpy as np
from datetime import datetime

from src.agents import BeliefLearner, RandomAgent
from src.environment import FootballEnv

ALPHA_DECAY = 0.9999954  # Decays alpha from 1.0 to ~0.01 over 1,000,000 steps
DEFAULT_SEEDS = [10, 20, 30, 42, 88]

def save_results_in_json(results: dict):
    filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {filename}")

def train_agents(playerA, playerB, env: FootballEnv, num_steps: int = 1_000_000):
    state = env.reset()
    for _ in tqdm(range(num_steps), desc="Training Progress", unit="step", leave=False):
        actionA = playerA.choose_action(state)
        actionB = playerB.choose_action(state)
        next_state, rewardA, rewardB, done = env.action(actionA, actionB)

        playerA.update(state, actionA, actionB, rewardA, rewardB, next_state, done)
        playerB.update(state, actionB, actionA, rewardB, rewardA, next_state, done)

        playerA.alpha *= ALPHA_DECAY
        playerB.alpha *= ALPHA_DECAY

        state = env.reset() if done else next_state

def test_agents(playerA, playerB, env: FootballEnv, num_steps: int = 100_000) -> dict:
    playerA.epsilon = 0.0  # Disable exploration for evaluation
    playerB.epsilon = 0.0

    state = env.reset()
    total_rewardA = 0
    total_rewardB = 0
    num_wins_A = 0
    num_wins_B = 0
    num_draws = 0
    done_count = 0

    for _ in tqdm(range(num_steps), desc="Testing Progress", unit="step", leave=False):
        actionA = playerA.choose_action(state)
        actionB = playerB.choose_action(state)
        next_state, rewardA, rewardB, done = env.action(actionA, actionB)

        total_rewardA += rewardA
        total_rewardB += rewardB

        if done:
            done_count += 1
            if rewardA > 0:
                num_wins_A += 1
            elif rewardB > 0:
                num_wins_B += 1
            else:
                num_draws += 1
            state = env.reset()
        else:
            state = next_state

    perc_wins_A = (num_wins_A / done_count * 100) if done_count > 0 else 0.0
    perc_wins_B = (num_wins_B / done_count * 100) if done_count > 0 else 0.0
    perc_draws = (num_draws / done_count * 100) if done_count > 0 else 0.0

    return {
        "n": done_count,
        "rew_A": total_rewardA,
        "rew_B": total_rewardB,
        "%_A": perc_wins_A,
        "%_B": perc_wins_B,
        "%_draw": perc_draws,
    }

def run_single_seed(seed: int, train_steps: int = 1_000_000, test_steps: int = 100_000) -> dict:
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(6)

    # 1) Belief Learner vs Random Agent
    env_rand = FootballEnv(width=5, height=4, rng=np.random.default_rng(child_seeds[0]))
    playerA_vs_random = BeliefLearner(rng=np.random.default_rng(child_seeds[1]))
    random_agent = RandomAgent(rng=np.random.default_rng(child_seeds[2]))

    train_agents(playerA_vs_random, random_agent, env_rand, num_steps=train_steps)
    random_metrics = test_agents(playerA_vs_random, random_agent, env_rand, num_steps=test_steps)

    # 2) Belief Learner vs Belief Learner (Identical Self-Play)
    env_ident = FootballEnv(width=5, height=4, rng=np.random.default_rng(child_seeds[3]))
    playerA = BeliefLearner(rng=np.random.default_rng(child_seeds[4]))
    playerB = BeliefLearner(rng=np.random.default_rng(child_seeds[5]))

    train_agents(playerA, playerB, env_ident, num_steps=train_steps)
    identical_metrics = test_agents(playerA, playerB, env_ident, num_steps=test_steps)

    return {
        "seed": seed,
        "random": random_metrics,
        "identical": identical_metrics,
    }

def aggregate_metrics(per_seed_results: list[dict], mode: str) -> dict:
    keys = ["n", "rew_A", "rew_B", "%_A", "%_B", "%_draw"]
    summary = {}
    for k in keys:
        vals = np.array([r[mode][k] for r in per_seed_results], dtype=float)
        summary[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }
    return summary

def main():
    message = """
    ======================================================================================
    ** Multi-Agent Reinforcement Learning in Football Environment **
    ======================================================================================
    This program simulates a football environment where two agents, Player A and Player B, 
    learn to play football using reinforcement learning techniques.
    
    List of the test:
    1) Random Learning:    Belief Learner vs Random Agent
    2) Identical Learning: Belief Learner vs Belief Learner
    
    """
    print(message)
    per_seed_results = []
    for seed in DEFAULT_SEEDS:
        print(f"--- Seed {seed} ---")
        res = run_single_seed(seed=seed, train_steps=1_000_000, test_steps=100_000)
        per_seed_results.append(res)
        print(
            f"  [vs Random]    Matches: {res['random']['n']} | "
            f"A: {res['random']['%_A']:.2f}% | B: {res['random']['%_B']:.2f}% | Draw: {res['random']['%_draw']:.2f}%"
        )
        print(
            f"  [vs Identical] Matches: {res['identical']['n']} | "
            f"A: {res['identical']['%_A']:.2f}% | B: {res['identical']['%_B']:.2f}% | Draw: {res['identical']['%_draw']:.2f}%"
        )

    summary = {
        "random": aggregate_metrics(per_seed_results, "random"),
        "identical": aggregate_metrics(per_seed_results, "identical"),
    }

    print("\n" + "=" * 86)
    print("AGGREGATED SUMMARY (Mean ± Std across seeds)")
    print("=" * 86)
    for mode, label in [("random", "Belief Learner vs Random Agent"), ("identical", "Belief Learner vs Belief Learner")]:
        s = summary[mode]
        print(
            f"{label}:\n"
            f"  Win A : {s['%_A']['mean']:.2f}% ± {s['%_A']['std']:.2f}%\n"
            f"  Win B : {s['%_B']['mean']:.2f}% ± {s['%_B']['std']:.2f}%\n"
            f"  Draws : {s['%_draw']['mean']:.2f}% ± {s['%_draw']['std']:.2f}%"
        )

    save_results_in_json({
        "seeds": DEFAULT_SEEDS,
        "summary": summary,
        "per_seed": per_seed_results,
    })

if __name__ == "__main__":
    main()