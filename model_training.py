import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

labels = pd.read_pickle('labels.pkl')
features = pd.read_pickle('features.pkl')
res = pd.concat([pd.Series(v, name=k) for k, v in features.items()], axis=1)
df = pd.concat([res, labels], axis=1)
df.dropna(how='all', inplace=True, thresh=100)
df.dropna(how='all', inplace=True, axis=1)

# encode emotion feature
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

dummy_data = pd.get_dummies(df['emotion'])
dummy_data.rename(columns=emotion_dict, inplace=True)
df = pd.concat([df, dummy_data], axis=1)
df.drop(['emotion'], axis=1, inplace=True)

X = df.loc[:, :'mouth_point_19_y']
y = df.loc[:, 'Angry':]

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# from sklearn.svm import SVC # "Support Vector Classifier"
# clf = SVC(kernel='rbf',verbose=True)
# clf.fit(X_train,y_train)

# build the network
# create model
num_classes = 7
model = Sequential()

model.add(Dense(X_train.shape[1] // 2,
                activation='sigmoid'))

model.add(Dense(X_train.shape[1] // 4,
                activation='sigmoid'))

model.add(Dense(num_classes,
                activation='softmax'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
EPOCHES = 5
BATCH_SIZE = 32
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=EPOCHES,
                    batch_size=BATCH_SIZE,
                    verbose=1)

# Visualize loss and accuracy
import matplotlib.pyplot as plt

accuracy = history.history['acc']
val_accuracy = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, label='Training accuracy')
plt.plot(epochs, val_accuracy, label='Validation accuracy')
plt.xlabel('epoches')
plt.ylabel('accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.xlabel('epoches')
plt.ylabel('loss')
plt.title('Training and validation loss')
plt.interactive(True)
plt.plot()
plt.show(block=True)
# model evaluation
test_eval = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
# save the model
model.save('emotion_model.h5')
