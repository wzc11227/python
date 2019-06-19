import numpy as np
import pandas as pd
import os
import time

np.random.seed(2)

N_STATES = 21
ACTIONS = ['1', '2', '3', '4', '5']
EPSILON = 0.9
ALPHA = 0.1
LAMBDA = 1
MAX_EPISODES = 10000
FRESH_TIME = 0.01


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions,
    )
    # print(table)
    return table


def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()
    return action_name


def get_env_feedback(S, A):
    if int(S) + int(A) == N_STATES:
        S_ = 'terminal'
        R = 1
    elif int(S) + int(A) > N_STATES:
        S_ = 'terminal'
        R = -1
    else:
        S_ = int(S) + int(A)
        R = 0
    return S_, R


def rl():
    if os.path.exists('rl.csv'):
        q_table = pd.read_csv('rl.csv')
    else:
        q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        S1last = 0
        S2last = 0
        winner = 0
        is_terminated = False
        while not is_terminated:
            if step_counter%2 == 0:
                A1 = choose_action(S, q_table)
                S_1, R1 = get_env_feedback(S, A1)
                print("trainer1:" + str(A1) + "\tsum:" + str(S_1))
                q1_predict = q_table.ix[S, A1]
                if S_1 != 'terminal':
                    S1last = S
                    q1_target = R1 + LAMBDA * (-q_table.iloc[S_1, :].max())
                else:
                    q1_target = R1
                    is_terminated = True
                    winner = 1 if (R1==1) else 2
                q_table.ix[S, A1] += ALPHA * (q1_target - q1_predict)
                S = S_1
            else:
                A2 = choose_action(S, q_table)
                S_2, R2 = get_env_feedback(S, A2)
                print("trainer2:" + str(A2) + "\tsum:" + str(S_2))
                q2_predict = q_table.ix[S, A2]
                if S_2 != 'terminal':
                    S2last = S
                    q2_target = R2 + LAMBDA * (-q_table.iloc[S_2, :].max())
                else:
                    q2_target = R2
                    is_terminated = True
                    winner = 2 if (R2 == 1) else 1
                q_table.ix[S, A2] += ALPHA * (q2_target - q2_predict)
                S = S_2
            step_counter += 1
        print("winner:" + ("trainer1" if(winner==1) else "trainer2"))
        if winner == 1:
            q_table.ix[S2last, A2] += ALPHA * (-1 - q_table.ix[S2last, A2])
        else:
            q_table.ix[S1last, A1] += ALPHA * (-1 - q_table.ix[S1last, A1])
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
    q_table.to_csv('rl.csv', encoding='utf-8', index=False)
