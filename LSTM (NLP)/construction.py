
import numpy as np
import collections
from tensorflow.keras.preprocessing.sequence import pad_sequences

class construction():
    """Contruction of word dictionary mapping for embedding layers
        paras:
        word_dict: trained word dictionary, mapping word to a unique vector
        word_list: a list of each sample that contains a list of words 
    """
    
    def __init__(self,word_dict,word_list,dim,id1,id2):
        """
        id1,id2: index list of separated set
        dim: embedding vector dimension, determined by word2vec model
        """
        
        self.word_dict=word_dict
        self.word_list=word_list
        
        self.id1=id1
        self.id2=id2
        
        self.vocab = list(word_dict.keys())
        self.embd= list(word_dict.values())
        self.vocab_size = len(self.vocab)
        
        print("Attention! Your dim should be consistent with your word2vec model!")
        self.embedding_dim =dim
    
    
    def dictionary(self):
        """Build dictionary and reversed dictionary to map integer to word
            The integer will become the input in the layer later
        """
        train_set=[self.word_list[i] for i in self.id1]
        training=[]
        for i in train_set:
            training+= i
    
        count = collections.Counter(training).most_common() 
        #creates list of word/count pairs
        #if the word appears less than two times we ignore it 
        count=[count[i] for i in range(len(count)) if count[i][1]>1]

        self.dictionary1 = {new_list: "" for new_list in range(1,len(count)+1)} 

        i=0
        for word, _ in count:
            self.dictionary1[i]=word
            i+=1
    
        #dictionary[word] = len(dictionary)+1 #len(dictionary) increases each iteration    
        self.dictionary2 = dict(zip(self.dictionary1.values(), self.dictionary1.keys()))
        print("\n Dictionary construction finished")
        
        return self.dictionary1,self.dictionary2
    
    
    def pad(self,max_doc_len):
        """
        Pad or truncate sentence to the same length
        """
        #max_doc_len:maximum resume words length for a certain company
        
        len_doc=[len(i) for i in self.word_list]
        int_map_list=['NA']*len(self.word_list)
        
        for i in range(len(self.word_list)):
            words=[self.word_list[i][j] for j in range(len_doc[i])]
            int_map=[self.dictionary2.get(k,'NA') for k in words]
            #remove NA because they represent the word occurs less than 3 times
            int_map=list(filter(('NA').__ne__, int_map))
            int_map_list[i]=int_map
        
        # Padding the words into same length
        padded= pad_sequences(int_map_list, maxlen=max_doc_len, padding='post')
            
        self.train_int=np.array([padded[i] for i in self.id1[3000:]])
        self.valid_int=np.array([padded[i] for i in self.id1[0:3000]])
        self.test_int=np.array([padded[i] for i in self.id2])
        
        
        return self.train_int,self.valid_int,self.test_int

   
    def embedding_word(self):
        """
        Generate the embedding weights for lstm layer,which is exactly training data vectors
        """
        doc_size=len(self.dictionary1)
        dict_as_list = sorted(self.dictionary2.items(), key = lambda x : x[1])
        embeddings_tmp=['NA']*(doc_size+1)
        
        #this return the vector embeddings of training data in an ordered array
        for i in range(doc_size+1):
            #print (100*i/doc_size,'%')
            if i!=doc_size:
                item = dict_as_list[i][0]
                embeddings_tmp[i]=self.word_dict.get(item,np.random.uniform(low=-0.2, high=0.2,size=self.embedding_dim))
            else:
                embeddings_tmp[i]=np.random.uniform(low=-0.2, high=0.2,size=self.embedding_dim)
        
        self.embeddings=np.array(embeddings_tmp)
        print("\n Training data embedding finished")
        
        return self.embeddings

######################################################################################
Embed=construction(word_dict,word_list,60,id1,id2)
dict1,dict2=Embed.dictionary()

keys1= list(dict1.keys())
keys2= list(dict2.keys())
print("dictionary sample: ", {key:dict1[key] for key in keys1[0:5]})
print("reverse dictionary sample: ", {key:dict2[key] for key in keys2[0:5]})
print("Total length of dictionary: ",len(dict1))

train_int,valid_int,test_int=Embed.pad(400)
embeddings=Embed.embedding_word()



