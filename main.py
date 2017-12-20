import numpy
from torch.autograd import Variable
from flask import Flask, jsonify, render_template, request
from PIL import Image
import torch
from sklearn.externals import joblib


# webapp
app = Flask(__name__)

def predict_with_pretrain_model(sample): 
    """
    	'''
	Args:
		sample: A integer ndarray indicating an image, whose shape is (28,28).

	Returns:
		A list consists of 10 double numbers, which denotes the probabilities of numbers(from 0 to 9).
		like [0.1,0.1,0.2,0.05,0.05,0.1,0.1,0.1,0.1,0.1].
	'''
    """
    net = joblib.load('LeNet5.pkl') 
    inputs = Variable(torch.from_numpy(sample).float())/255 # wrap them in Variable 
    outputs = net.forward(inputs) # forward   
    result = numpy.exp(outputs.data)/(numpy.exp(outputs.data)).sum() # softmax
    return result.tolist()[0]

@app.route('/api/mnist', methods=['POST'])
def mnist():
    input = ((numpy.array(request.json, dtype=numpy.uint8))).reshape(1,1,28, 28)
    output = predict_with_pretrain_model(input)
    return jsonify(results=output)


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
