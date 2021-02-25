from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, Input, Average
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint


class Model_architecture():

    def fit_model(self, train_generator, val_generator, epochs, checkpoint_model_name):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(28, 28, 1),
                         data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=5, strides=2, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Conv2D(64, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=5, strides=2, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        model.add(Dense(10, activation='softmax'))

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
        mc = ModelCheckpoint(checkpoint_model_name, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

        adam = optimizers.Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                      optimizer=adam)

        history = model.fit(train_generator,
                            epochs=epochs,
                            validation_data=val_generator,
                            verbose=2,
                            callbacks=[es, mc])
        return model, history


    def fit_ensemble(self, models, train_generator):
        input_shape = (28, 28, 1)
        model_input = Input(shape=input_shape)

        merged_models = list()
        for model in models:
            model.trainable = False
            merged_models.append(model(model_input))

        y = Average()(merged_models)

        model = Model(model_input, y, name='ensemble')
        adam = optimizers.Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                      optimizer=adam)

        history = model.fit(train_generator,
                            epochs=1,
                            verbose=2)
        return model, history