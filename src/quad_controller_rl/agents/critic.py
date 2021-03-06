"""Deep Deterministec Policy Gradients (DDPG) reinforcement learning agent."""
from keras import layers, models, optimizers
from keras import backend as K
from keras.regularizers import l2

class Critic:
    """Critic (Value) Model. """

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        =========
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values"""
        l2_lambda = 0.001
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        states_bn = layers.BatchNormalization()(states)
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=4, activation=None, kernel_initializer='he_uniform', kernel_regularizer=l2(l2_lambda))(states_bn)
        net_states = layers.Dropout(0.1)(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation('relu')(net_states)
        net_states = layers.Dense(units=4, activation=None, kernel_initializer='he_uniform', kernel_regularizer=l2(l2_lambda))(net_states)
        net_states = layers.Dropout(0.1)(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation('relu')(net_states)
        net_states = layers.Dense(units=8, activation=None, kernel_initializer='he_uniform', kernel_regularizer=l2(l2_lambda))(net_states)
        net_states = layers.Dropout(0.1)(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation('relu')(net_states)
        net_states = layers.Dense(units=8, activation=None, kernel_initializer='glorot_normal', kernel_regularizer=l2(l2_lambda))(net_states)
        net_states = layers.Dropout(0.1)(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation('relu')(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=4, activation=None, kernel_initializer='he_uniform', kernel_regularizer=l2(l2_lambda))(actions)
        net_actions = layers.Dropout(0.1)(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation('relu')(net_actions)
        net_actions = layers.Dense(units=4, activation=None, kernel_initializer='he_uniform', kernel_regularizer=l2(l2_lambda))(net_actions)
        net_actions = layers.Dropout(0.1)(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation('relu')(net_actions)
        net_actions = layers.Dense(units=8, activation=None, kernel_initializer='he_uniform', kernel_regularizer=l2(l2_lambda))(net_actions)
        net_actions = layers.Dropout(0.1)(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation('relu')(net_actions)
        net_actions = layers.Dense(units=8, activation=None, kernel_initializer='glorot_normal', kernel_regularizer=l2(l2_lambda))(net_actions)
        net_actions = layers.Dropout(0.1)(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation('relu')(net_actions)

        # Try different layer sizes, activations, add batch norm, regularizers, etc.

        # Combine state and ation pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to produce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used
        # by an actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)