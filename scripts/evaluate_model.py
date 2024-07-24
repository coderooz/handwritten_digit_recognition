import tensorflow as tf
from data.mnist_data import load_data

def evaluate_model():
    (x_train, y_train), (x_test, y_test) = load_data()
    model = tf.keras.models.load_model('cnn_model.h5')
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc:.4f}')

if __name__ == '__main__':
    evaluate_model()
