#Analisa a quantidade de VP,FP,VN e FN e calcula as métricas
class confusionMatrix:
    def __init__(self,test,pred,beta=1):
        self.VP,self.FP,self.VN, self.FN = 0,0,0,0
        self.size = len(pred)
        self.analysis(pred,test)
        self.accuracy = self.accuracy()
        self.precision = self.precision()
        self.recall = self.recall()
        self.npv = self.NPV()
        self.f_score = self.f_score(beta)

    def analysis(self,pred,test):
        for i in range(0,len(pred)):
          if(pred[i]==0 and test[i]==0):
            self.VN=self.VN+1
          if(pred[i]==1 and test[i]==0):
            self.FP=self.FP+1
          if(pred[i]==0 and test[i]==1):
            self.FN=self.FN+1
          if(pred[i]==1 and test[i]==1):
            self.VP=self.VP+1
    def accuracy(self):
        return (self.VP+self.VN)/self.size
    def precision(self):
        return (self.VP)/(self.VP + self.FP)
    def recall(self):
        return (self.VP)/(self.VP + self.FN)
    def NPV(self):
        return (self.VN)/(self.VN + self.FN)
    def f_score(self,beta):
        return (1+pow(beta,2))*(self.precision*self.recall)/(pow(beta,2)*self.precision + self.recall)