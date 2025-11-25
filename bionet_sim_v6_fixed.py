import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import requests
import re

# BioNet-Sim v6: Multi-Task SNN with Emergence Demo
# ------------------------------------------------
# Bio-inspiration:
# 1. External Input Plasticity: Adapts to unstructured web data via repurposed circuits.
# 2. Social Learning: Merge for collaborative knowledge.
# 3. Lifelong Learning: Sequential tasks without forgetting.
# 4. Emergence: In MountainCar, expect "swing-building" to emerge from prior skills.

class LIFNeuron:
    def __init__(self, n_inputs, n_hidden):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.W = torch.randn(n_inputs, n_hidden) * 0.05
        self.reset()

    def reset(self):
        self.v = torch.zeros(self.n_hidden)
        self.trace_pre = torch.zeros(self.W.shape[0])
        self.trace_post = torch.zeros(self.n_hidden)
        self.eligibility = torch.zeros_like(self.W)

    def forward(self, x_in):
        current = torch.matmul(x_in, self.W)
        self.v = self.v * 0.9 + current
        spikes = (self.v >= 1.0).float()
        self.v = self.v * (1 - spikes)
        
        self.trace_pre = self.trace_pre * 0.95 + x_in
        self.trace_post = self.trace_post * 0.95 + spikes
        
        ltp = torch.outer(self.trace_pre, spikes)
        ltd = torch.outer(x_in, self.trace_post)
        self.eligibility += ltp - ltd * 0.5
        
        return spikes

    def save(self, filename):
        data = {
            'W': self.W.tolist(),
            'eligibility': self.eligibility.tolist(),
            'n_inputs': self.n_inputs,
            'n_hidden': self.n_hidden
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f"Brain saved to {filename}")

    def load(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        self.W = torch.tensor(data['W']).float()
        self.eligibility = torch.tensor(data['eligibility']).float()
        self.n_inputs = data['n_inputs']
        self.n_hidden = data['n_hidden']
        print(f"Brain loaded from {filename}")

def merge_neurons(n1, n2, alpha=0.5):
    print("Merging brains (Social Learning)...")
    merged = LIFNeuron(n1.n_inputs, n1.n_hidden)
    merged.W = alpha * n1.W + (1 - alpha) * n2.W
    merged.eligibility = alpha * n1.eligibility + (1 - alpha) * n2.eligibility
    return merged

def preprocess_obs(obs, env_name, target_size=1600):
    if 'MountainCar' in env_name:
        bounds = np.array([1.2, 0.07])  # Pos -1.2 to 0.6, Vel -0.07 to 0.07 (norm to -1 to 1)
        obs = np.clip(obs / bounds, -1, 1)
        padded = np.zeros(target_size)
        padded[:len(obs)] = obs
        return padded
    else:
        # Fallback for CartPole
        bounds = np.array([2.4, 3.0, 0.21, 3.0])
        obs = np.clip(obs / bounds, -1, 1)
        padded = np.zeros(target_size)
        padded[:len(obs)] = obs
        return padded

def poisson_encoding(obs, max_rate_hz=100.0, dt=0.01):
    rates = np.abs(obs) * max_rate_hz
    prob = rates * dt
    spikes = (np.random.rand(len(obs)) < prob).astype(float)
    return torch.tensor(spikes).float()

# --- Internet Hook Functions ---

def fetch_text_data(url):
    try:
        print(f"Fetching data from {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to fetch: {response.status_code}")
            return ""
    except Exception as e:
        print(f"Error fetching: {e}")
        return ""

def encode_text_to_spikes(text, input_size=1600):
    text = re.sub(r'\W+', ' ', text.lower())
    words = text.split()
    pos_words = ['good', 'great', 'best', 'excellent', 'love', 'happy', 'win', 'success']
    neg_words = ['bad', 'worst', 'hate', 'sad', 'fail', 'lose', 'terrible', 'awful']
    pos_count = sum(1 for w in words if w in pos_words)
    neg_count = sum(1 for w in words if w in neg_words)
    pos_rate = min(pos_count / len(words), 1.0) if words else 0.0
    neg_rate = min(neg_count / len(words), 1.0) if words else 0.0
    half = input_size // 2
    spikes = np.zeros(input_size)
    spikes[:half] = np.random.rand(half) < pos_rate
    spikes[half:] = np.random.rand(input_size - half) < neg_rate
    return torch.tensor(spikes).float()

def train_web_classifier(snn):
    print("\n=== Phase 5: Build Sentiment Classifier from Web ===")
    samples = [
        {'url': 'https://en.wikipedia.org/wiki/Happiness', 'label': 1},  # Pos
        {'url': 'https://en.wikipedia.org/wiki/Love', 'label': 1},         # Pos
        {'url': 'https://en.wikipedia.org/wiki/Sadness', 'label': 0},     # Neg
        {'url': 'https://en.wikipedia.org/wiki/Anger', 'label': 0},       # Neg
    ]
    lr = 0.005
    correct_count = 0
    for i, sample in enumerate(samples):
        text = fetch_text_data(sample['url'])
        if not text: continue
        x_in = encode_text_to_spikes(text, snn.n_inputs)
        output_spikes = torch.zeros(snn.n_hidden)
        for _ in range(10):
            spikes = snn.forward(x_in)
            output_spikes += spikes
        half = snn.n_hidden // 2
        neg_activity = output_spikes[:half].sum()
        pos_activity = output_spikes[half:].sum()
        prediction = 1 if pos_activity > neg_activity else 0
        reward = 1.0 if prediction == sample['label'] else -1.0
        if prediction == sample['label']: correct_count += 1
        snn.W += lr * reward * snn.eligibility
        snn.W = torch.clamp(snn.W, -2.0, 2.0)
        print(f"Sample {i+1}: URL={sample['url']} | Pred={prediction} | Label={sample['label']} | Reward={reward}")
    accuracy = correct_count / len(samples) * 100
    print(f"Web Training Accuracy: {accuracy:.1f}%")
    snn.save('bionet_web_trained.json')

def train_full_pipeline():
    max_inputs = 1600
    n_hidden = 256
    # Phase 1: CartPole
    print("\n=== Phase 1: Agent A learns CartPole ===\n")
    agent_a = LIFNeuron(max_inputs, n_hidden)
    train_task(agent_a, 'CartPole-v1', max_inputs, episodes=10, render=False)
    # Phase 2 & 3: Merge
    print("\n=== Phase 2 & 3: Social Merging ===\n")
    agent_b = LIFNeuron(max_inputs, n_hidden)
    agent_shared = merge_neurons(agent_a, agent_b, alpha=0.7)
    # Phase 4: MountainCar (Physics)
    print("\n=== Phase 4: Shared Brain learns MountainCar (Physics) ===\n")
    train_task(agent_shared, 'MountainCar-v0', max_inputs, episodes=20, render=False)
    # Phase 5: Web Training
    train_web_classifier(agent_shared)
    print("\nBioNet-Sim v6 Complete.")

def train_task(snn, env_name, input_size, episodes, render=False):
    try:
        render_mode = 'human' if render else None
        env = gym.make(env_name, render_mode=render_mode)
    except: return
    lr = 0.002
    epsilon = 1.0
    epsilon_decay = 0.95
    epsilon_min = 0.05
    rewards = []
    for episode in range(episodes):
        obs, _ = env.reset() if hasattr(env, 'reset') and env.reset.__code__.co_argcount > 0 else (env.reset(), {})
        if isinstance(obs, tuple): obs = obs[0]
        snn.reset()
        total_reward = 0
        done = False
        while not done:
            processed = preprocess_obs(obs, env_name, input_size)
            x_in = poisson_encoding(processed)
            spikes = snn.forward(x_in)
            if np.random.rand() < epsilon: action = env.action_space.sample()
            else:
                n_actions = env.action_space.n
                group_size = snn.n_hidden // n_actions
                activity = [spikes[i*group_size:(i+1)*group_size].sum() for i in range(n_actions)]
                action = np.argmax(activity)
            step_result = env.step(action)
            if len(step_result) == 5: obs, reward, terminated, truncated, _ = step_result; done = terminated or truncated
            else: obs, reward, done, _ = step_result
            total_reward += reward
        baseline = np.mean(rewards[-10:]) if len(rewards) > 0 else -200.0 if 'MountainCar' in env_name else 10.0
        snn.W += lr * (total_reward - baseline) * snn.eligibility
        snn.W = torch.clamp(snn.W, -2.0, 2.0)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards.append(total_reward)
        print(f"[{env_name}] Ep {episode+1} | Reward: {total_reward:.1f}")
    env.close()

if __name__ == "__main__":
    train_full_pipeline()
