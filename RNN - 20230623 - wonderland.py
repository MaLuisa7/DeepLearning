import numpy as np
import pandas as pd
from keras.layers import LSTM, SimpleRNN, Dense, Activation
from keras.models import Sequential
path = "C:/Users/Usuario/Documents/Deep Learning/wonderland.txt"

#Extract the text
print('Extracting text from input ...')
fin = open(path, 'rb')
lines =[]
for line in fin:
    line = line.strip().lower()
    line = line.decode("ascii", "ignore")
    if len(line) ==0:
        continue
    lines.append(line)
fin.close()
text = " ".join(lines)

#creating lookup tables
# here chars is the number of features in our character "vocabulary"
chars = set([c for c in text])
nb_chars = len(chars)
char2index = dict((c,i) for i, c in enumerate(chars))
index2char = dict((i,c) for i, c in enumerate(chars))

#create inputs and labels from the text. We do this by stepping
# through the text $step$ character at a time, and extracting a
#sequence of size $seqlen$ and the next output char. For example,
#assuming an imput text @The sky was falling@, we would get the
# following sequence of inputs_chars and labels_chars (first 5 only)
#example
#the sky wa -->s
#he sky was -->
#e sky was  -->f
# sky was f -->a
#sky was fa -->l

print("creating tinput and label text")

SEQLEN = 10
STEP = 1

input_chars  =[]
label_chars = []
for i in range( 0 ,len(text) - SEQLEN , STEP):
    input_chars.append((text[i:i + SEQLEN]))
    label_chars.append(text[i + SEQLEN])


# Vectorize the input and label chars
# Each row of the inputs is represented by seqlen characters, each
# represented as a 1-hot encoding of size len(char). There are
# len(inputs_chars) such rows,
# so shape(X) is (len(input_chars), seqlen, nbchars)

print("Vectorizing inputs and label text")
X = np.zeros((len(input_chars), SEQLEN, nb_chars), dtype=  bool)
y = np.zeros((len(input_chars), nb_chars), dtype=  bool)
for i, input_char in enumerate(input_chars):
    for j ,ch in enumerate(input_char):
        X[i,j,char2index[ch]] = 1
    y[i, char2index[label_chars[i]]] = 1

X.shape. y.shape
y[0,]# esta posicionada en el antepenultimo
X[0,] #posicion13 esta la letra


#Build model . We use a single RNN with a fully connected layer
# to compute the most likely predicted output char

HIDDEN_SIZE = 128
BATCH_SIZE = 128
NUM_ITERATIONS = 25
NUM_EPOCHS_PER_ITERATION = 1
NUM_PREDS_PER_EPOCH = 100

model = Sequential()
model.add(SimpleRNN(HIDDEN_SIZE, return_sequences=False, input_shape = (SEQLEN, nb_chars),
                    unroll=True))
model.add(Dense(nb_chars))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

# We train the model in batches and test putput generated at each step

for iteration in range(NUM_ITERATIONS):
    print("="*50)
    print("Iteration #: %d" % (iteration))
    model.fit(X,y,batch_size=BATCH_SIZE, epochs=NUM_EPOCHS_PER_ITERATION)

    #testing model
    #randomly choose a row from input_chars, then use it to
    #generate text from model for next 100 chars

    test_idx = np.random.randint(len(input_chars))
    test_chars = input_chars[test_idx]
    print("Generating from seed : %s" %(test_chars))
    print(test_chars, end = "")
    for i in range(NUM_PREDS_PER_EPOCH):
        Xtest = np.zeros((1, SEQLEN, nb_chars))
        for i ,ch in enumerate(test_chars):
            Xtest[0, i, char2index[ch]] = 1
        pred = model.predict(Xtest, verbose=0)[0]
        ypred = index2char[np.argmax(pred)]
        print(ypred, end="")
        #move forward with test_chars + ypred
        test_chars = test_chars[1:] + ypred
    print()
