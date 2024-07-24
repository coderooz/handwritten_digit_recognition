from data.mnist_data import load_data
from model.cnn_model import create_model

def train_model():
    (x_train, y_train), (x_test, y_test) = load_data()
    model = create_model()
    model.fit(x_train, y_train, epochs=5, validation_split=0.1)
    model.save('cnn_model.h5')

if __name__ == '__main__':
    train_model()
