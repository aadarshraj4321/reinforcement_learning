import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO

class HumanFeedbackEnv(gym.Env):
    def __init__(self):
        super(HumanFeedbackEnv, self).__init__()
        
        self.action_space = spaces.Discrete(10)
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        

        self.responses = [
            "Hello! It's a pleasure to meet you. How can I assist you today?", 
            "What's on your mind? Is there something specific you'd like to know about?", 
            "The weather outside is quite unpredictable, isn't it? Let's talk about your favorite topics.", 
            "Do you have any hobbies you enjoy? Or perhaps you'd like to learn something new today?", 
            "Goodbye for now! Feel free to return whenever you have more questions.", 
            "I believe that learning something new every day is key to success. What do you think?", 
            "Life is full of unexpected turns. It's always a good idea to plan ahead, isn't it?", 
            "How can I assist you with any challenges you're facing at the moment?", 
            "Did you know that the universe is expanding faster than we once thought? Fascinating, isn't it?", 
            "Goodbyes are always hard, but remember that each goodbye leads to a new beginning."
        ]
        
    
        self.feedback = {
            "good": set(),
            "bad": set()
        }
        
    def reset(self):
        """Reset the environment and return the initial observation."""
        return np.random.rand(10)
        
    def step(self, action):
        """Perform an action and return the new state, reward, done, and info."""
        action = int(action)  
        response = self.responses[action]
        reward = 0
        
        if response in self.feedback["good"]:
            reward = 1
        elif response in self.feedback["bad"]:
            reward = -1
        
        obs = np.random.rand(10)
        done = True 
        
        return obs, reward, done, {}
    
    def render(self, mode='human'):
        """Rendering is not required for this environment."""
        pass
    
    def provide_feedback(self, response, is_good):
        """Provide feedback on a response."""
        if is_good:
            if response in self.feedback["bad"]:
                self.feedback["bad"].remove(response)  # Remove from "bad" if marked as "good"
            self.feedback["good"].add(response)
        else:
            if response in self.feedback["good"]:
                self.feedback["good"].remove(response)  # Remove from "good" if marked as "bad"
            self.feedback["bad"].add(response)


env = HumanFeedbackEnv()

model = PPO("MlpPolicy", env, verbose=1) 
model.learn(total_timesteps=20000)

model.save("ppo_human_feedback_training")


def test_and_provide_feedback(env, model): # test the model and collect feedback
    obs = env.reset()
    print("Testing and providing feedback:\n")
    
    for i in range(30):
        action, _states = model.predict(obs)
        action = int(action) 
        
        obs, reward, done, _ = env.step(action)
        response = env.responses[action] 
        
        print(f"Test {i+1}: Response: '{response}', Reward: {reward}")
        
        feedback = input("Was this response good? (yes/no): ").strip().lower()
        if feedback == "yes":
            env.provide_feedback(response, is_good=True)  
        elif feedback == "no":
            env.provide_feedback(response, is_good=False) 
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
        
       
        model.learn(total_timesteps=1000)   # Retrain the model with new feedback # Adjust timesteps as needed


test_and_provide_feedback(env, model) # Test the model and provide feedback


def test_updated_model(env, model): # test the updated model to see changes
    obs = env.reset()
    print("Testing the updated model:\n")
    
    for i in range(10):
        action, _states = model.predict(obs)
        action = int(action) 
        
        obs, reward, done, _ = env.step(action)
        response = env.responses[action] 
        
        print(f"Updated Test {i+1}: Response: '{response}', Reward: {reward}")



test_updated_model(env, model) # Test the updated model
