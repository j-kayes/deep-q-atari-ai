'''
    Agent learns to beat the Atari games from OpenAI gym.
    Copywrite James Kayes (c) 2019
'''
import numpy as np
import random
import keras
import keras.backend as K
import cv2
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from statistics import mean, median

class Agent:

    def __init__(self, environment, state_size, learning_rate=5e-6,
                trace_size=4, memory_size=1000000):
        self.env = environment
        self.trace_size = trace_size
        self.state_size = state_size

        self.action_space = np.zeros(shape=(self.env.action_space.n, self.env.action_space.n))
        n = 0
        for n in range(self.env.action_space.n):
            self.action_space[n][n] = 1

        self.memory = []
        self.memory_size = memory_size
        self.memory_index = 0

        self.lr = learning_rate

        # Input to the model is a sequence of scaled down screen states:
        self.input_shape = (84, 84, trace_size)

        self.build_model()

    def build_model(self, drop_rate=0.0):
        self.x_input = Input(shape=self.input_shape, name='x')

        hidden_1 = keras.layers.Conv2D(16, (8,8), strides=(4, 4), activation='relu')(self.x_input)
        hidden_2 = keras.layers.Conv2D(32, (4,4), strides=(2, 2), activation='relu')(hidden_1)

        flattened = keras.layers.Flatten()(hidden_2)
        hidden_3 = keras.layers.Dense(256, activation='relu')(flattened)

        # Different output for each action:
        self.y_outputs = []
        for action_q in range(self.env.action_space.n):
            self.y_outputs.append(Dense(1)(hidden_3))

        self.model = Model(inputs=self.x_input, outputs=self.y_outputs)

        self.model.compile(loss='mean_squared_error',
                    loss_weights=[1.0 for n in range(self.env.action_space.n)],
                    metrics=['accuracy'],
                    optimizer=Adam(lr=self.lr))

    def process_sequence(self, sequence_data):
        # Down samples and processes a sequence of data in a memory trace of the
        # size needed by the model.
        sequence_trace = sequence_data[-self.trace_size:]
        sequence_diff = self.trace_size - len(sequence_trace)
        if(sequence_diff > 0):
            for i in range(sequence_diff):
                # Duplicate last state:
                sequence_trace.append(sequence_trace[-1])

        sequence_trace = np.array(sequence_trace)
        # Remove colour:
        sequence_trace = np.mean(sequence_trace, axis=3, keepdims=False)
        # Down sample:
        downsampled_trace = np.empty((84,84,len(sequence_trace)))
        index = 0
        for state_image in sequence_trace:
            resized = cv2.resize(state_image, (84, 110), interpolation=cv2.INTER_AREA)
            downsampled_trace[:, :, index] = resized[18:102, :]
            index += 1

        return np.array(downsampled_trace).astype(np.uint8)

    def get_output(self, x_input):
        return self.model.predict(x_input)

    def get_best_q_value(self, x_input, batch_index=0):
        full_output = self.model.predict(x_input)

        best_q = None
        for action_q_batch in full_output:
            if(best_q is None):
                best_q = action_q_batch[batch_index]
            elif(action_q_batch > best_q):
                best_q = action_q_batch[batch_index]

        return best_q

    def get_best_action(self, x_input, batch_index=0):
        # List of q-values for each action:
        full_output = self.model.predict(x_input)

        action_n = 0
        best_action = None
        best_q = None
        for output_q_batches in full_output:
            if(best_q is None):
                best_q = output_q_batches[batch_index]
                best_action = action_n
            elif(output_q_batches[batch_index] > best_q):
                best_q = output_q_batches[batch_index]
                best_action = action_n

            action_n += 1

        return best_action

    def get_samples(self, stop_after_limit=True, n_games=500000, max_t=250000,
                    epsilon=1.0, display_frames=False):
        # Will append the memory with state/action/reward data and return the
        # average score:
        game_counter = 0
        scores = []
        frames = 0
        for game in range(n_games):
            game_counter += 1
            initial_state = self.env.reset()
            sequence = []  # Sequence of states
            sequence.append(initial_state)
            score = 0.0
            # Auto-reset after this:
            for t in range(max_t):
                frames += 1
                action = None
                processed_current_state = np.array([self.process_sequence(sequence)])
                assert(processed_current_state.shape is not (1,84,84,4))
                # We start with a high epsilon:
                if(random.random() < epsilon):
                    action = self.env.action_space.sample()
                else:
                    # Best action for this processed sequence:
                    action = self.get_best_action(processed_current_state)

                # Get the reward/state information after taking this action:
                next_state, reward, done, infom = self.env.step(action)
                if(display_frames):
                    self.env.render()
                score += reward
                if(not done):
                    sequence.append(next_state)
                processed_next_state = self.process_sequence(sequence)
                # Append to memory (up to limit):
                if(len(self.memory) < self.memory_size):
                    self.memory.append((processed_current_state, action, reward, processed_next_state, done))
                elif(self.memory_index < self.memory_size):
                    # Overwrite from the beginning (when full):
                    self.memory[self.memory_index] = (processed_current_state, action, reward, processed_next_state, done)
                    self.memory_index += 1
                else:
                    self.memory[0] = (processed_current_state, action, reward, processed_next_state)
                    self.memory_index = 1

                if(done):  # Game over
                    break
            scores.append(score)

            if(stop_after_limit):
                if(len(self.memory) >= self.memory_size):
                    break
        return mean(scores), frames

    def train_network(self, target_mean_score=1250.0, games=10000000, batch_size=32,
            initial_epsilon=1.0, final_epsilon=0.1, epsilon_frames_range=1000000,
            gamma=0.95, score_sample_size=250, train_every=20, batches_per_train=150):
        # Trains the agent.
        # Play randomly until memory has at least batch_size entries
        while(len(self.memory) < batch_size):
            self.get_samples(False, 1, epsilon=1.0)
        total_frames = 0
        total_batch_updates = 0
        scores = []
        for game in range(games):
            # Scales linearly from initial to final epsilon:
            if(total_frames < epsilon_frames_range):
                e = initial_epsilon - ((initial_epsilon - final_epsilon) * (total_frames / epsilon_frames_range))
            else:
                e = final_epsilon
            # Play a  game, and record data to memory buffer:
            score, frames = self.get_samples(False, 1, epsilon=e)
            scores.append(score)
            if(len(scores) == score_sample_size):
                # Removing first element after a maximum sample size reached.
                scores = scores[1:]

            m_score = mean(scores)
            total_frames += frames

            if((m_score >= target_mean_score) and (len(scores) > 200)):
                print('Target mean score reached')
                break
            if(game % train_every == 0 and (game is not 0)):
                # Sample and train on mini-batch:
                mini_batch = random.sample(self.memory, batches_per_train*batch_size)
                p_state_batches = []
                # Each action has its own batch:
                q_batches = [[] for x in range(self.env.action_space.n)] 
                for p_state, action, reward, p_next_state, done in mini_batch:
                    p_state_batches.append(p_state[0])
                    p_state = np.array([p_state])
                    p_next_state = np.array([p_next_state])
                    target = reward
                    if(not done):
                        target = reward + gamma*self.get_best_q_value(p_next_state)

                    # Predicted Q-values for each action according to the model:
                    q_values = self.model.predict(p_state[0])
                    q_values[action] = np.array([target]) # With updated target.

                    for a in range(len(q_values)):
                        q_batches[a].append(q_values[a][0])

                # Model expects a list of np arrays:
                q_batches = [np.array(q_batch) for q_batch in q_batches] # q_batch for each action.

                display_progress = False
                if(game % (5*train_every) == 0): # Only print training progress every 5 updates:
                    display_progress = True
                    self.model.save('latest_save.h5')
                if(game % (10*train_every) == 0): # Every 10th update:
                    filename = str(game) + "-" + str(m_score) + "-" + str(total_frames) + "-" + str(total_batch_updates)
                    self.model.save(filename + ".h5")

                # Try to fit to the target Q-values for each action:
                self.model.fit(np.array(p_state_batches), q_batches, batch_size=batch_size, verbose=int(display_progress))

                total_batch_updates += batches_per_train
                print('Games: {}, Mean Score: {:.2f}, Frames: {}, e: {:.4f}, Batch Updates: {}'.format(
                game, m_score, total_frames, e, total_batch_updates))

        print('Training complete')
        self.model.save('final_model.h5')
        print('Model saved: final_model.h5')

    def load_model(self, path):
        print('Attempting to load previously saved model...')
        return load_model(path)

    # Play and record scores with the currently loaded graph:
    def play(self, games=1, epsilo=0.05, show_game=False):
        scores = []
        total_frames = 0
        for game in range(games):
            game_num = game + 1
            score, frames = self.get_samples(False, 1, epsilon=epsilo, display_frames=show_game)
            scores.append(score)
            m_score = mean(scores)
            med_score = median(scores)

            total_frames += frames
            print('Games: {} Score: {}, Mean Score: {:.2f}, Median: {:.2f}, Total Frames: {}'.format(
                    game_num,
                    score,
                    m_score,
                    med_score,
                    total_frames))
