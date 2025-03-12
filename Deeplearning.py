import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, GRU, Dense, Dropout

def create_dl_model(layer_type='LSTM', num_layers=3, layer_sizes=[50, 50, 50], input_shape=(60, 1), dropout_rate=0.2):
    """
    Creates a Deep Learning model dynamically based on input parameters.
    
    Parameters:
    - layer_type (str): Type of recurrent layer ('LSTM', 'RNN', 'GRU').
    - num_layers (int): Number of layers in the model.
    - layer_sizes (list): List of neuron sizes for each layer.
    - input_shape (tuple): Shape of the input data (timesteps, features).
    - dropout_rate (float): Dropout rate to prevent overfitting.
    
    Returns:
    - model (Sequential): Compiled deep learning model.
    """
    model = Sequential()
    
    # Select the correct layer type
    layer_dict = {'LSTM': LSTM, 'RNN': SimpleRNN, 'GRU': GRU}
    if layer_type not in layer_dict:
        raise ValueError("Invalid layer type. Choose from 'LSTM', 'RNN', or 'GRU'.")
    Layer = layer_dict[layer_type]
    
    # Add first layer with input shape
    model.add(Layer(units=layer_sizes[0], return_sequences=(num_layers > 1), input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    
    # Add hidden layers
    for i in range(1, num_layers - 1):
        model.add(Layer(units=layer_sizes[i], return_sequences=True))
        model.add(Dropout(dropout_rate))
    
    # Add final recurrent layer without return_sequences
    if num_layers > 1:
        model.add(Layer(units=layer_sizes[-1]))
        model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(units=1))
    
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Experimenting with Different Architectures
lstm_model = create_dl_model(layer_type='LSTM', num_layers=3, layer_sizes=[50, 50, 50])
rnn_model = create_dl_model(layer_type='RNN', num_layers=3, layer_sizes=[40, 40, 40])
gru_model = create_dl_model(layer_type='GRU', num_layers=3, layer_sizes=[60, 60, 60])

lstm_model.summary()
rnn_model.summary()
gru_model.summary()