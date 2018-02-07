import matplotlib.pyplot as plt
from tensorflow.python import keras
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping

import numpy as np


class Seq2Seq:
    def __init__(self, decode_max_len, start_seq_index, input_bits, output_bits):
        # creating the model
        self.input_bits = input_bits
        self.output_bits = output_bits
        self.encoded_bits = 256
        self.maxlen = decode_max_len
        self.start_seq_index = start_seq_index

        print("-- decode_max_len --")
        print(decode_max_len)

        print("-- start_seq_index --")
        print(start_seq_index)

        print("-- decode_max_len --")
        print(decode_max_len)

        print("-- input_bits --")
        print(input_bits)

        print("-- output_bits --")
        print(output_bits)

        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.input_bits))
        print(encoder_inputs)

        encoder = LSTM(self.encoded_bits, return_state=True)
        print(encoder)

        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        print(encoder_outputs)

        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.output_bits))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.encoded_bits, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

        print("dense layer : " + str(self.output_bits))
        decoder_dense = Dense(self.output_bits, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.trainingModel = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.trainingModel.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        # Inference setup:

        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(self.encoded_bits,))
        decoder_state_input_c = Input(shape=(self.encoded_bits,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    def trainModel(self, encoder_input_data, decoder_input_data, decoder_target_data, batch_size=10, epochs=1):
        earlyStopping = EarlyStopping(patience=50, monitor='loss', mode='auto', min_delta=0)
        tensorBoard = keras.callbacks.TensorBoard(log_dir='tblogs', histogram_freq=1, write_graph=True,
                                                  write_grads=True, write_images=False)

        hist = self.trainingModel.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                                      batch_size=batch_size,
                                      epochs=epochs, callbacks=[earlyStopping])

        print(hist.history['loss'])

        # 모델 세이브
        self.trainingModel.save('s2s.h5')

        # hist 를 이용하여 loss 그래프 그리기
        # self.historyLossGraph(hist)

    def historyLossGraph(self, hist):
        loss_ax = plt.subplot()
        loss_ax.plot(hist.history['loss'], 'y', label='train loss')
        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')

        loss_ax.legend(loc='upper left')
        plt.show()

    def loadWeights(self):
        self.trainingModel.load_weights('s2s.h5')

    def setInput(self, input_data):
        # Encode the input as state vectors.
        self.states_value = self.encoder_model.predict(input_data)
        self.target_seq = np.zeros((1, 1, self.output_bits))
        self.target_seq[0, 0, self.start_seq_index] = 1

    def predictNext(self):
        output, h, c = self.decoder_model.predict([self.target_seq] + self.states_value)

        # Sample a token
        output_index = np.argmax(output[0, -1, :])

        self.target_seq = np.zeros((1, 1, self.output_bits))
        self.target_seq[0, 0, output_index] = 1

        # Update states
        self.states_value = [h, c]
        return output[0, -1, :]
