'''
    Agent learns to beat the Atari games from OpenAI gym.
    Copywrite James Kayes (c) 2019
'''
import tensorflow as tf
import numpy as np
import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from statistics import mean, median


class Agent:

    def __init__(self, environment, state_size, learning_rate=1e-4, sequence_length=10, memory_size=100000):
        self.env = environment
        self.sequence_length = sequence_length
        self.state_size = state_size

        self.action_space = np.zeros(shape=(self.env.action_space.n, self.env.action_space.n))
        n = 0
        for n in range(self.env.action_space.n):
            self.action_space[n][n] = 1

        # Input size will be the size of the previous sequence buffer, which
        # includes a sample of states and inputs up to the the most recent:
        self.input_size = sequence_length*(self.env.action_space.n + self.state_size) - self.env.action_space.n

        self.memory = []
        self.memory_size = memory_size
        self.memory_index = 0

        self.lr = learning_rate

        self.build_model()
        # For saving and loading the graph after training:
        self.saver = tf.train.Saver(save_relative_paths=True)


    def __enter__(self):
        return self

    # TODO: Has hold rate changed to dropout?
    def build_model(self, hidden_layers=4, layer_connections=128, hold_rate=0.5):
        # First dimension is the batch size:
        self.x_input = Dense(layer_connections, input_shape=(self.input_size,), activation='relu')

        for layer in range(hidden_layers-1):
            if(layer is 0):
                hidden_layer = Dense(layer_connections, activation='relu')(self.x_input)
            else:
                hidden_layer = Dense(layer_connections, activation='relu')(hidden_layer)
            dropout = Dropout(hold_rate)(hidden_layer)

        self.y_outputs = Dense(self.env.action_space.n)  # Output layer.

        # TODO: Model only allows 1 sample for now:
        self.action_q_values = [q_value for q_value in self.y_outputs[0]]

        self.model = Model(inputs=self.x_input, outputs=self.action_q_values)
        #self.model.summary()
        self.model.compile(loss='mean_squared_error', loss_weights=[1.0 for i in range(self.env.action_space.n)], metrics=['accuracy'], optimizer='adam')

    def process_sequence(self, sequence_data):
        # Pass in the full list of sequence data and this will preprocess into
        # the p_buffer, so that it can be fed to the graph.

        # Processed sequence needs to be the same size as the input:
        p_size = self.input_size
        p_buffer = np.zeros(shape=p_size)
        i = 0
        # Loop backwards through p_buffer:
        while(i < p_size):
            if(i < len(sequence_data)):
                i += 1
                p_buffer[p_size - i] = sequence_data[len(sequence_data) - i]
            else:
                # For the sake of simplicity, I will fill the remaining buffer
                # with 0's:
                for index in range(p_size - i):
                    p_buffer[index] = 0
                i = p_size

        return p_buffer.reshape((-1, len(p_buffer)))

    def get_best_action(self, x_input):
        output = self.model.predict(x_input)
        return np.argmax(output, axis=1)[0]

    def get_output(self, x_input):
        return self.model.predict(x_input)

    def get_best_q_value(self, x_input, batch_index=0):
        q_value = np.max(self.model.predict(x_input), axis=1)
        return q_value[batch_index]

    def get_samples(self, stop_after_limit=True, n_games=500000, max_t=250, epsilon=1.0, display_frames=False):
        # Will append the memory with state/action/reward data and return the
        # average score across games:
        game_counter = 0
        scores = []
        frames = 0
        for game in range(n_games):
            game_counter += 1
            initial_state = self.env.reset()
            sequence = []  # Sequence of states
            sequence.extend(initial_state)
            score = 0.0
            # Auto-reset after this:
            for t in range(max_t):
                frames += 1
                action = None
                processed_current_state = self.process_sequence(sequence)
                # We start with a high epsilon:
                if(random.random() < epsilon):
                    action = self.env.action_space.sample()
                else:
                    # Best action for this processed sequence(acording to the
                    # model):
                    action = self.get_best_action(processed_current_state)
                # Get the reward/state information after taking this action:
                next_state, reward, done, infom = self.env.step(action)
                if(display_frames):
                    self.env.render()
                score += reward
                if(not done):
                    sequence.extend(np.concatenate((self.action_space[action], next_state)).tolist())
                processed_next_state = self.process_sequence(sequence)
                # Append to memory (up to limit):
                if(len(self.memory) < self.memory_size):
                    self.memory.append((processed_current_state, action, reward, processed_next_state, done))
                elif(self.memory_index < self.memory_size):
                    # Overwrite from the beginning(when full):
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

    def train_network(self, target_mean_score=250.0, games=1000, batch_size=32, initial_epsilon=1.0, final_epsilon=0.05, epsilon_frames_range=30000, gamma=0.95, score_sample_size=250):
        # Trains the agent.
        # Play randomly until memory has at least batch_size entries
        while(len(self.memory) < batch_size):
            self.get_samples(False, 1, epsilon=1.0)
        total_frames = 0
        scores = []
        for game in range(games):
            # Scales linearly from initial to final epsilon up to epsilon
            # frames range:
            if(total_frames < epsilon_frames_range):
                e = initial_epsilon - ((initial_epsilon - final_epsilon) * (total_frames / epsilon_frames_range))
            else:
                e = final_epsilon
            # Play a random game, and record data to memory buffer:
            score, frames = self.get_samples(False, 1, epsilon=e)
            scores.append(score)
            m_score = mean(scores)
            if(len(scores) == score_sample_size):
                # Removing first element after a maximum sample size reached.
                scores = scores[1:]

            total_frames += frames
            print('Game: {} Score: {}, Mean: {:.2f} Frames: {} e: {:.4f}'.format(game, score, m_score, total_frames, e))

            if((m_score >= target_mean_score) and (len(scores) > 100)):
                print('Target mean score reached')
                break
            # Sample and train on mini-batch:
            mini_batch = random.sample(self.memory, batch_size)
            for p_state, action, reward, p_next_state, done in mini_batch:
                target = reward
                if(not done):
                    target = reward + gamma*self.get_best_q_value(p_next_state)

                # Train:
                self.model.fit(p_state, np.array([target_q_values]), verbose=0)

        print('Training complete')
        save_path = self.saver.save(self.sess, 'saved/model.ckpt')
        print('Model saved: {}'.format(save_path))

    def load_model(self, path):
        print('Attempting to load previously saved model...')
        self.saver.restore(self.sess, path)

    # Play and record scores with the currently loaded graph:
    def play(self, games=100, epsilo=0.05, show_game=True):
        scores = []
        total_frames = 0
        for game in range(games):
            game_num = game + 1
            score, frames = self.get_samples(False, 1, epsilon=epsilo, display_frames=show_game)
            scores.append(score)
            m_score = mean(scores)
            med_score = median(scores)

            total_frames += frames
            print('Games: {} Score: {}, Mean Score: {:.2f}, Median: {:.2f} Total Frames: {}'.format(
                    game_num,
                    score,
                    m_score,
                    med_score,
                    total_frames))
