from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping


class Neural_Network():
    def __init__(self,embeddings,max_doc_len):
        """
        doc_size: total number of appeared words
        embeddings: embeddings acquired by construction module
        max_doc_len: maximun sentence length, must be same as before
        """
        keras.backend.clear_session()
        self.embeddings=embeddings
        self.doc_size=len(dict1)
        self.max_doc_len=max_doc_len
        
    def build(self,units,dense,dropout):
        """
        units: units of LSTM layer
        dense: units of first dense layer
        dropout: dropout proportion, decimal between 0 and 1
        """
        self.model = Sequential()
        self.model.add(Embedding(self.doc_size+1, 60, weights=[self.embeddings],input_length=self.max_doc_len,mask_zero=True))
        #model.add(Embedding(doc_size+1, 60,input_length=max_doc_len,mask_zero=True))
        self.model.add(Dropout(dropout))
        #model.add(LSTM(64,input_shape=(max_doc_len,60),return_sequences=True))
        self.model.add(LSTM(units,input_shape=(self.max_doc_len,60)))
        self.model.add(Dense(dense,activation="relu"))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(4, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
        print(self.model.summary())
        
        return self.model
    
    def fit(self,train_x,validation_x,train_y,validation_y,logdir):
        """
        All the input x should be in the form of integer mappings of word
        """
        callback = EarlyStopping(monitor='val_loss', patience=5)
        tb_cb = keras.callbacks.TensorBoard(log_dir=logdir)
        
        result=self.model.fit(train_x,train_y, epochs=20, batch_size=128,callbacks=[callback,tb_cb],validation_data=(validation_x, validation_y))
        history = pd.DataFrame(result.history) 

        return history

    def prediction():
        predicty=self.model.predict(test_int)
        
        return predicty
        
##################################################################################################

NN=Neural_Network(embeddings,400)
model=NN.build(64,32,0.1)
fitness=NN.fit(train_int,valid_int,train_y,valid_y,'/home/niyu/Documents/Project/Lab/code')

prediction=NN.prediction()


df_prediction=pd.DataFrame([[prediction,test_y]])


from sklearn.metrics import confusion_matrix
prediction=list(prediction)
typelist1=[]
typelist2=[]

for i in range(len(test_y)):
    pre=list(test_y[i])
    ture=list(test_y[i])
    typelist2.append(pre.index(max(pre)))
    typelist1.append(ture.index(max(ture)))


con=confusion_matrix(typelist1, typelist2)


