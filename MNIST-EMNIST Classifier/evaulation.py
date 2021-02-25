from utility-apis import UtilityAPIs


class Evaluation:
    utilities = None
    def evaluate_mnist(self, model, data, label):
        print("Evaluate mnist")

        self.utilities.get_mnist_data()
        _, acc = model.evaluate(data, label)
        print("Accuracy on data: %.3f" % acc)


    def evaluate_emnist(self, model, data, label):
        print("Evaluate emnist")

        self.utilities.get_emnist_data()
        _, acc = model.evaluate(data, label)
        print("Accuracy on data: %.3f" % acc)


    def evaluate(self, model, data, label):
        self.utilities = UtilityAPIs()
        self.evaluate_mnist(model, data, label)
        self.evaluate_emnist(model, data, label)
