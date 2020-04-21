import music21
import pathlib
import pickle

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


# 输入 100 个音符，输出之后的一个音符
input_notes_length = 100

output_dir = 'Bach'


def get_all_notes():
    midi_dir = pathlib.Path('./res/')
    all_midi_files = midi_dir.glob('**/*.mid')

    print("{} file(s) found.".format(len(list(midi_dir.glob('**/*.mid')))))
    all_notes = []

    for i, midi_file in enumerate(all_midi_files):
        midi = music21.converter.parse(midi_file)
        # midi = midi[0]

        try:
            s2 = music21.instrument.partitionByInstrument(midi)
            notes = s2.parts[0].recurse()
        except:
            notes = midi.flat.notes

        # 把和弦转换成 str
        for element in notes:
            if isinstance(element, music21.note.Note):
                all_notes.append(str(element.pitch))
            elif isinstance(element, music21.chord.Chord):
                all_notes.append('.'.join(str(n) for n in element.normalOrder))

        print("\r{} file(s) written.".format(i + 1), end="")

    with open('output/{}/all_notes'.format(output_dir), 'wb') as f:
        pickle.dump(all_notes, f)

    vocab = sorted(set(all_notes))
    print("\nvocab's length: ", len(vocab))
    print("#notes: ", len(all_notes))
    return all_notes, vocab


def make_seq(all_notes, vocab):
    # 建立 one-hot 词典
    note_dict = {}
    for i, note in enumerate(vocab):
        note_dict[note] = i

    num_training = len(all_notes) - input_notes_length
    input_notes_in_vocab = np.zeros((num_training, input_notes_length, len(vocab)))
    output_notes_in_vocab = np.zeros((num_training, len(vocab)))

    for i in range(num_training):
        input_notes = all_notes[i: i + input_notes_length]
        output_note = all_notes[i + input_notes_length]
        for j, note in enumerate(input_notes):
            input_notes_in_vocab[i, j, note_dict[note]] = 1
        output_notes_in_vocab[i, note_dict[output_note]] = 1
        print("\r{} / {}".format(i+1, num_training), end="")
    print()
    return input_notes_in_vocab, output_notes_in_vocab


def build_network(num_vocab):
    model = keras.Sequential([
        # keras.layers.LSTM(128, return_sequences=True,
        #                   input_shape=(input_notes_length, num_vocab)),
        # keras.layers.Dropout(0.2),
        # keras.layers.LSTM(128, return_sequences=False),
        # keras.layers.Dropout(0.2),
        # keras.layers.Dense(num_vocab, activation='softmax')
        keras.layers.LSTM(512, recurrent_dropout=0.3, return_sequences=True,
                          input_shape=(input_notes_length, num_vocab)),
        keras.layers.LSTM(512),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_vocab, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    return model


def train(model, input_notes_in_vocab, output_notes_in_vocab):
    callback = keras.callbacks.ModelCheckpoint(
        filepath='output/' + output_dir + '/cp-{epoch:03d}.ckpt',
        verbose=1,
        save_weights_only=True,
        period=5
    )
    history = model.fit(input_notes_in_vocab, output_notes_in_vocab,
                        batch_size=128, epochs=200, callbacks=[callback])
    return history


if __name__ == '__main__':
    all_notes, vocab = get_all_notes()
    inputs, outputs = make_seq(all_notes, vocab)
    model = build_network(len(vocab))
    train(model, inputs, outputs)
    model.save_weights('output/{}/weights.h5'.format(output_dir))

