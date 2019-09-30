from gensim.models import Word2Vec

class word2vec():
    # word embedding for word_list
    
    def __init__(self,corpus=None):
        #word_list: list of list with splited words
        self.corpus=corpus
        
        
    def embedding(self):
        """
        Word2Vec parameters:
        min_count: minimun occurance of word 
        size: word vector size
        workers: number of partitions during training 
        window: window size 
        sg: The training algorithm, either CBOW(0) or skip gram(1)
        """
        if self.corpus !=None:
            model = Word2Vec(self.corpus, min_count=3,size= 50,workers=3, window =3, sg = 1)
            print("Training Embedding finished")
        else:
            model= Word2Vec.load("/home/niyu/Documents/Project/Lab/Word60.model")
            print("Loading Embedding finished")
        return model
    
    def word_dict(self):
        #convert embedding list to dict
        model=self.embedding()
        word_dict = dict({})

        for idx, key in enumerate(model.wv.vocab):
            word_dict[key] = model.wv[key]   
        return word_dict
        

##############################################################################
w2v=word2vec()
word_dict=w2v.word_dict()

