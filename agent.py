'''
    Agent learns to beat the Atari games from OpenAI gym.
    Copywrite James Kayes (c) 2019
'''
import tensorflow as tf
import numpy as np
import random
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from statistics import mean, median


def cust_loss(y_true, y_pred):
    return (y_true[0] - y_pred[0])**2

class Agent:

    def __init__(self, environment, state_size, learning_rate=1e-4,
    sequence_length=10, memory_size=100000):
        self.env = environment
        self.sequence_length = sequence_length
        self.state_size = state_size

        self.action_space = np.zeros(shape=(self.env.action_space.n, self.env.action_space.n))
        n = 0
        for n in range(self.env.action_space.n):
            self.action_space[n][n] = 1

        # Input size will be the size of the previous sequence buffer which
        # includes a sample of states and inputs up to the the most recent:
        self.input_size = sequence_length*(self.env.action_space.n + self.state_size) - self.env.action_space.n

        self.memory = []
        self.memory_size = memory_size
        self.memory_index = 0

        self.lr = learning_rate

        self.build_model()
        # For saving and loading the graph after training:
        self.saver = tf.train.Saver(save_relative_paths=True)

    def build_model(self, hidden_layers=5, layer_connections=128, drop_rate=0.0):
        # First dimension is the batch size:
        self.x_input = Input(shape=(self.input_size,), name='y')

        for layer in range(hidden_layers):
            if(layer is 0):
                hidden_layer = Dense(layer_connections, activation='relu')(self.x_input)
            else:
                dropout = Dropout(drop_rate)(hidden_layer)
                hidden_layer = Dense(layer_connections, activation='relu')(dropout)

        # Different output for each action:
        self.y_outputs = []
        for action_index in range(self.env.action_space.n):
            self.y_outputs.append(Dense(1, activation='relu')(hidden_layer))

        self.model = Model(inputs=self.x_input, outputs=self.y_outputs)

        #self.model.summary()
        self.model.compile(loss=cust_loss,
        loss_weights=[1.0 for n in range(self.env.action_space.n)],
        metrics=['accuracy'],
        optimizer='sgd')
        K.set_value(self.model.optimizer.lr, 1e-4)


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
                # For the sake of simplicity, I will fill the remaining buffer with 0's:
                for index in range(p_size - i):
                    p_buffer[index] = 0
                i = p_size

        return p_buffer.reshape((-1, len(p_buffer)))

    def get_best_action(self, x_input, batch_index=0):
        # List of q-values for each action:
        full_output = self.model.predict(x_input)

        action_n = 0
        best_action = None
        best_q = None
        for output_batches in full_output:
            if(best_q is None):
                best_q = output_batches[batch_index]
                best_action = action_n
            elif(output_batches[batch_index] > best_q):
                best_q = output_batches[batch_index]
                best_action = action_n

            action_n += 1

        return best_action

    def get_output(self, x_input):
        return self.model.predict(x_input)

    def get_best_q_value(self, x_input, batch_index=0):
        full_output = self.model.predict(x_input)

        best_q = None
        for output_batches in full_output:
            if(best_q is None):
                best_q = output_batches[batch_index]
            elif(output_batches[batch_index] > best_q):
                best_q = output_batches[batch_index]

        return best_q

    def get_samples(self, stop_after_limit=True, n_games=500000, max_t=250,
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
                    # Best action for this processed sequence(acording to the model):
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

    def train_network(self, target_mean_score=245.0, games=1000, batch_size=64,
    initial_epsilon=1.0, final_epsilon=0.05, epsilon_frames_range=10000,
    gamma=0.95, score_sample_size=250):
        # Trains the agent.
        # Play randomly until memory has at least batch_size entries
        print("Training")
        while(len(self.memory) < batch_size):
            self.get_samples(False, 1, epsilon=1.0)
        total_frames = 0
        scores = []
        for game in range(games):
            # Scales linearly from initial to final epsilon up to epsilon frames range:
            if(total_frames < epsilon_frames_range):
                e = initial_epsilon - ((initial_epsilon - final_epsilon) * (total_frames / epsilon_frames_range))
            else:
                e = final_epsilon
            # Play a random game, and record data to memory buffer:
            score, frames = self.get_samples(False, 1, epsilon=e)
            scores.append(score)
            if(len(scores) == score_sample_size):
                # Removing first element after a maximum sample size reached.
                scores = scores[1:]

            m_score = mean(scores)
            total_frames += frames
            print('Game: {} Score: {}, Mean: {:.2f} Frames: {} e: {:.4f}'.format(
            game, score, m_score, total_frames, e))

            if((m_score >= target_mean_score) and (len(scores) > 100)):
                print('Target mean score reached')
                break
            # Sample and train on mini-batch:
            mini_batch = random.sample(self.memory, batch_size)
            p_batch = []
            target_q_batch = []
            for p_state, action, reward, p_next_state, done in mini_batch:
                target = reward
                if(not done):
                    target = reward + gamma*self.get_best_q_value(p_next_state)

                # Predicted Q-values for each action according to the model:
                q_values = self.model.predict(p_state, batch_size=1)
                actual = self.model.predict(p_state, batch_size=1)
                q_values[action] = np.array([target]) # With updated target.

                #print(K.get_value(self.model.optimizer.lr))

                # Try to fit to the target Q-values for each action:
                self.model.train_on_batch(p_state, q_values)

        print('Training complete')
        # TODO: Model saving:
        #save_path = self.saver.save(self.sess, 'saved/model.ckpt')
        #print('Model saved: {}'.format(save_path))

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
