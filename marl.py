import pandas as pd
import random
import os
import pickle
from openai import OpenAI
from env import CellTowerEnv
from agent import QLearningAgent


client = OpenAI(api_key='')

user = {0: 'max user limit not reached', 1: 'max user limit reached'}
coverage = {0: 'good coverage quality', 1:'fair coverage quality', 2:'poor coverage quality'}

def llm_action(ag1_state, min_p1, max_p1, ag2_state, min_p2, max_p2):
    prompt = f'''Neighbor cell tower state:
    power level: {min_p2 + ag2_state[1]} (min: {min_p2}, max: {max_p2})
    coverage: {ag2_state[2]} (0: good, 1: fair, 2: poor)
    user connection: {ag2_state[3]} (0: under max limit, 1: reached max limit)
    Target cell tower state:
    power level: {min_p1 + ag1_state[1]} (min: {min_p1}, max: {max_p1})
    coverage: {ag1_state[2]} (0: good, 1: fair, 2: poor)
    user connection: {ag1_state[3]}(0: under max limit, 1: reached max limit)
    Best action for the target cell tower:'''

    # prompt = f'''Target cell tower state:
    # power level: {min_p1 + ag1_state[1]} (min: {min_p1}, max: {max_p1})
    # coverage: {ag1_state[2]} (0: good, 1: fair, 2: poor)
    # user connection: {ag1_state[3]}(0: under max limit, 1: reached max limit)
    # Best action for the target cell tower:'''

    response = client.chat.completions.create(
    model='gpt-4o',
    messages=[
        {'role': 'system', 'content': '''You are a decision-making assistant for a cellular base station. You must choose an action that optimizes coverage, user connectivity, and power:
         1. 'increase_power' - If the signal strength is weak and capacity allows, increase transmission power.
         2. 'decrease_power' - If there is excessive interference or unnecessary power consumption, decrease transmission power.
         3. 'do_nothing' - If network parameters are stable, take no action.
         4. 'handoff_request' - If reached maximum capacity and a neighboring tower is available, initiate a handoff request.
         Always base your decision on key metrics such as power level, capacity, coverage state, friction of poor signal, and neighboring cell availability. Return only the number of selected action without additional explanation.'''},
        {'role': 'user', 'content': prompt}
    ])

    return int(response.choices[0].message.content[0])-1


def get_delayed_reward(hist, min_p, max_p):
    prompt = f'Cell tower states:\n'
    for idx, i in hist.items():
        prompt += f'''Episode {idx+1}:
        Failed Handoff: {i[2]}
        Start:
        - Power Level: {min_p+i[0][0]} (Range: {min_p}-{max_p})
        - User Connection: {i[0][1]} (0: under max limit, 1: reached max limit)
        - Fraction of Good Signal Strength: {i[0][2]}
        - Fraction of Good SNR: {i[0][3]}
        End:
        - Power Level: {min_p+i[1][0]} (Range: {min_p}-{max_p})
        - User Connection: {i[1][1]} (0: under max limit, 1: reached max limit)
        - Fraction of Good Signal Strength: {i[1][2]}
        - Fraction of Good SNR: {i[1][3]}'''

    response = client.chat.completions.create(
    model='gpt-4o',
    messages=[
        {'role': 'system', 
         'content': '''You are an intelligent and fair evaluation agent for a cell tower's performance. Based on the following metrics, assign a single numerical reward value. 
         Performance Metrics:
         1. Power Efficiency (Lower is better): Indicates power consumption.
         2. Number of Failed Handoffs (Lower is better): Reflects user connectivity stability.
         3. Fraction of Good Signal Strength (Higher is better): Represents network quality.
         4. Fraction of Good SNR (Higher is better): Reflects signal clarity.
         **Tip:** Exceeding or falling below capacity, or having handoff failure are strong indicators of unsatisfactory performance. But when it has available resources, it should use it to avoid unsuccessful handoffs.
         Reward Scale:
         - +2 (Excellent): No failed handoff in any episode. Balanced efficiency, signal strength, and SNR.
         - +1 (Good): Minor inefficiencies in some performance metrics, but overall stable performance.
         - 0 (Fair): Some noticeable issues in above metrics, but acceptable performance.
         - -1 (Suboptimal): A few failed handoffs when additional resources can be allocated, weak performance, or moderate inefficiencies .
         - -2 (Poor): Some failed handoffs, high power usage, or low signal/SNR
         Consider all listed key metrics and map the tower's performance to a proper reward. Respond with a single integer (-2, -1, 0, 1, or 2). No explanation is required.'''},
        {'role': 'user', 'content': prompt+'\nReward: '}
    ])
    return int(response.choices[0].message.content[:2].strip())*1.2


def train_q_learning(num_episodes, max_steps):
    env1 = CellTowerEnv(43, 46, 1.6, 5, 50)
    env2 = CellTowerEnv(30, 37, 0.1, 1.3, 30)
    
    agent1 = QLearningAgent(
        state_size=env1.observation_space_size,
        action_size=env1.action_space_size,
        learning_rate=0.05,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.05,
        table='ag1_qtable.npy'
    )
    agent2 = QLearningAgent(
        state_size=env2.observation_space_size,
        action_size=env2.action_space_size,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.01,
        table='ag2_qtable.npy'
    )
    
    ag1_hist = {}
    ag2_hist = {}
    rewards_hist1 = []
    rewards_hist2 = []
    env1_vals = []
    env2_vals = []
    
    for episode in range(num_episodes):
        ls1 = []
        ls2 = []
        ag1_state = list(env1.reset())
        total_reward1 = 0
        ag2_state = list(env2.reset())
        total_reward2 = 0
        ag1_failed = 0
        ag2_failed = 0

        ls1.append((ag1_state[1], ag1_state[3], ag1_state[4], ag1_state[5]))
        ls2.append((ag2_state[1], ag2_state[3], ag2_state[4], ag2_state[5]))
        agent1.hist.clear()
        agent2.hist.clear()

        
        for step in range(max_steps):
            # action_r1 = agent1.choose_action(ag1_state[0])
            llm_decision = llm_action(ag1_state, 43, 46, ag2_state, 30, 37)
            rl_decision = agent1.choose_action(ag1_state[0])
            action_r1 = agent1.make_decision(ag1_state[0], llm_decision, rl_decision)
            next_r1, reward1, sig1, snr1 = env1.step(action_r1)
            agent1.update_trust(ag1_state[0], rl_decision, reward1, llm_decision)
            agent1.learn(ag1_state[0], action_r1, reward1, next_r1)
            agent1.add_pair(ag1_state[0], action_r1)
            total_reward1 += reward1
            # if agent choose to handoff, hand over to the other agent
            if sig1 is not None:
                result1 = env2.add_request(sig1, snr1)
                if result1 == False:
                    ag1_failed += 1

            # action_r2 = agent2.choose_action(ag2_state[0])
            llm_decision2 = llm_action(ag2_state, 30, 37, ag1_state, 43, 46)
            rl_decision2 = agent2.choose_action(ag2_state[0])
            action_r2 = agent2.make_decision(ag2_state[0], llm_decision2, rl_decision2)
            next_r2, reward2, sig2, snr2 = env2.step(action_r2)
            agent2.update_trust(ag2_state[0], rl_decision, reward2, llm_decision2)
            agent2.learn(ag2_state[0], action_r2, reward2, next_r2)
            agent2.add_pair(ag2_state[0], action_r2)
            total_reward2 += reward2
            # if agent choose to handoff, hand over to the other agent
            if sig2 is not None:
                result2 = env1.add_request(sig2, snr2)
                if result2 == False:
                    ag2_failed += 1

            # simulate change in environment
            if step != (max_steps-1):
                ag1_state = list(env1.change_env())
                ag2_state = list(env2.change_env())

        # decay epsilon
        # agent1.epsilon = max(agent1.epsilon * agent1.epsilon_decay, agent1.epsilon_min)
        # agent2.epsilon = max(agent2.epsilon * agent2.epsilon_decay, agent2.epsilon_min)

        rewards_hist1.append(total_reward1)
        rewards_hist2.append(total_reward2)
        env1_vals.append([ag1_state[1]]+list(env1.get_values())+[ag1_failed])
        env2_vals.append([ag2_state[1]]+list(env2.get_values())+[ag2_failed])
        
        ls1 += [(ag1_state[1], ag1_state[3], ag1_state[4], ag1_state[5]), ag1_failed]
        ls2 += [(ag2_state[1], ag2_state[3], ag2_state[4], ag2_state[5]), ag2_failed]
        ag1_hist[episode] = ls1
        ag2_hist[episode] = ls2

        # get delayed reward every k episodes
        if (episode + 1) % 3 == 0:
            d_reward1  = get_delayed_reward(ag1_hist, 43, 46)
            if d_reward1 != 0:
                agent1.apply_delayed_reward(d_reward1)
            d_reward2  = get_delayed_reward(ag2_hist, 30, 37)
            if d_reward2 != 0:
                agent2.apply_delayed_reward(d_reward2)
            print(f'delayed rewards: {d_reward1} {d_reward2}')
            ag1_hist.clear()
            ag2_hist.clear()

        # print(f'''Episode {episode+1}/{num_episodes}\n
        #       Agent1 Total Reward: {total_reward1:.2f}, Trust: {agent1.trust_scores[ag1_state[0]]:.2f}
        #       Agent2 Total Reward: {total_reward2:.2f}, Trust: {agent2.trust_scores[ag2_state[0]]:.2f}''')
        # if (episode+1) % 100 == 0:
        #     print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent1.epsilon:.2f}")
        print(f'''Episode {episode+1}/{num_episodes}\n
              Agent1 Values: {env1_vals[-1]},
              Agent2 Values: {env2_vals[-1]}''')

    return agent1, rewards_hist1, env1_vals, agent2, rewards_hist2, env2_vals

# Run training
if __name__ == "__main__":
    # df = pd.read_csv('data.csv')
    ag1, r1, v1, ag2, r2, v2 = train_q_learning(100, 10)
    # print(sum(r1)/len(r1))
    # print(sum(r2)/len(r2))
    with open('marl_ag1.pkl', 'wb') as f1:
        pickle.dump(v1, f1)
    with open('marl_ag2.pkl', 'wb') as f2:
        pickle.dump(v2, f2)