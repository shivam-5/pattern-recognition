from keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
from UtilityAPIs import UtilityAPIs
from Evaluation import Evaluation
from Model_architecture import Model_architecture

utilities = UtilityAPIs()
eval = Evaluation()
arch = Model_architecture()

# Cross-validation
n_folds = 5
model_tuples = []

X_train, y_train, X_test, y_test = utilities.get_emnist_data()
# X_train, y_train, X_test, y_test = utilities.get_mnist_data()

datagenarator = utilities.augmentation()
test_generator = datagenarator.flow(X_test, y_test, batch_size=64)

path = './models/'
for i in range(n_folds):
    print("Training on Fold: ", i + 1)
    t_x, val_x, t_y, val_y = train_test_split(X_train, y_train, test_size=0.1,
                                              random_state=np.random.randint(1, 1000, 1)[0])
    train_generator = datagenarator.flow(t_x, t_y, batch_size=64)
    val_generator = datagenarator.flow(val_x, val_y, batch_size=64)

    checkpoint_name = path + 'checkpoint_model_' + str(i) + '.h5'
    model, model_history = arch.fit_model(train_generator, val_generator, epochs=70, checkpoint_model_name=checkpoint_name)
    model_tuples.append((model, model_history))

    model_name = path + 'final_model_' + str(i) + '.h5'
    model.save(model_name)

checkpoint_models = list()
for i in range(n_folds):
    checkpoint_models.append(load_model(path + 'checkpoint_model_' + str(i) + '.h5'))

model, model_history = arch.fit_ensemble(checkpoint_models, train_generator)

print("Evaluate training set on model...")
eval.evaluate(model, X_train, y_train)
print("Evaluate testing set on model...")
eval.evaluate(model, X_test, y_test)
