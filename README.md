# CNN-based-sentimental-analysis-multiclass-classification

This is a simple implementation of a CNC architecture made for a text classification problem.
The architecture is defined as following:
- An embedding layer used to provide a dense representation of words and their relative meanings.
- 3 convolutional layers used as N-gram filters with N={2, 3, 4} to capture richers correlations between words,
- each of the convolutional layers are followed by a max pooling layer to reduce the amount of parameters and computation in the network.
- a dense layer as a feedforward neural network
- followed by another dense layer for label probabiity prediction.

The model is applied on a dataset containing +1500000 tweets classified as negative or positive and has achieved 86,28% accuracy on training data and 82,32% accuracy on validation data.

Dataset used for the application: http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
Link to the paper: https://arxiv.org/pdf/1703.03091.pdf
