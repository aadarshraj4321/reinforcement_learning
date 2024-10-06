import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO

class HumanFeedbackEnv(gym.Env):
    def __init__(self):
        super(HumanFeedbackEnv, self).__init__()
        
        # Action space: 10 possible responses
        self.action_space = spaces.Discrete(10)
        
        # Observation space: A 10-dimensional vector between 0 and 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        
        # Responses for the agent to choose from
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
        
        # Feedback storage
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
            self.feedback["good"].add(response)
        else:
            self.feedback["bad"].add(response)

# Initialize the environment
env = HumanFeedbackEnv()

# Load the trained model
model = PPO.load("ppo_human_feedback_training")

def predict_new_response(env, model, obs):
    """Use the trained model to predict the best response for a given observation."""
    action, _states = model.predict(obs)
    action = int(action)
    response = env.responses[action]
    return response

def use_model(env, model):
    obs = env.reset()
    print("Chatbot is now online. Type 'exit' to end the chat.\n")
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        
        # Use the model to generate a response
        response = predict_new_response(env, model, obs)
        
        print(f"Chatbot: {response}")
        
        # Collect feedback from the user
        feedback = input("Was this response good? (yes/no): ").strip().lower()
        if feedback == "yes":
            env.provide_feedback(response, is_good=True)
        elif feedback == "no":
            env.provide_feedback(response, is_good=False)
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
        
        # Update observation
        obs = env.reset()

# Use the trained model
use_model(env, model)
