'''
    Agent learns to beat the Atari games from OpenAI gym.
    Copywrite James Kayes (c) 2019
'''
import numpy as np
import random
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from statistics import mean, median

def squared_difference(y_true, y_pred):
    return (y_true[0] - y_pred[0])**2

    #TODO: Test this:
def bilinear_interpolate(im, new_width, new_height):
        x = [v for v in range(new_width)]
        y = [v for v in range(new_height)]

        x = np.asarray(x)
        y = np.asarray(y)

        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1

        x0 = np.clip(x0, 0, im.shape[1]-1);
        x1 = np.clip(x1, 0, im.shape[1]-1);
        y0 = np.clip(y0, 0, im.shape[0]-1);
        y1 = np.clip(y1, 0, im.shape[0]-1);

        Ia = im[ y0, x0 ]
        Ib = im[ y1, x0 ]
        Ic = im[ y0, x1 ]
        Id = im[ y1, x1 ]

        wa = (x1-x) * (y1-y)
        wb = (x1-x) * (y-y0)
        wc = (x-x0) * (y1-y)
        wd = (x-x0) * (y-y0)

        return wa*Ia + wb*Ib + wc*Ic + wd*Id

class Agent:

    def __init__(self, environment, state_size, learning_rate=1e-4,
    trace_size=10, memory_size=100000, input_x=64, input_y=84):
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
        self.input_shape = (trace_size, input_y, input_x)

        self.build_model()

    def build_model(self, hidden_layers=5, layer_connections=128, drop_rate=0.0):
        # First dimension is the batch size:
        self.x_input = Input(shape=self.input_shape, name='y')

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
        self.model.compile(loss=squared_difference,
        loss_weights=[1.0 for n in range(self.env.action_space.n)],
        metrics=['accuracy'],
        optimizer='adam')
        K.set_value(self.model.optimizer.lr, self.lr)

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
        sequence_trace = np.mean(sequence_trace, axis=3)
        print("Seq:" + str(sequence_trace.shape))
        # Down sample:
        # Could also try tensorflow image downsampling:
        downsampled_trace = []
        for state_image in sequence_trace:
            downsampled_trace.append(bilinear_interpolate(state_image, 84, 84))
        downsampled_trace = np.array(downsampled_trace)
        print("Downsampled: " + str(downsampled_trace.shape))

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
            sequence.append(initial_state)
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
                    sequence.append(next_state)
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

    def train_network(self, target_mean_score=245.0, games=250, batch_size=32,
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
