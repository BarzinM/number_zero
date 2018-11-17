
class RunningAverage(object):
    def __init__(self, ratio=.1, initial=0.):
        self.avg = initial
        self.ratio = ratio

    def setRatio(self,ratio):
        self.ratio = ratio

    def average(self, value):
        self.avg = self.ratio * value + (1. - self.ratio) * self.avg
        return self.avg

class RunningSelection(object):
    def __init__(self,ratio=.1,initial_avg=0., initial_covar=0.):
        self.avg = initial_avg
        self.covar = initial_covar
        self.ratio = ratio

    def select(self,value,bound=.9):
        self.avg = self.ratio*value + (1.-self.ratio)*self.avg
        diff = abs(value-self.avg)
        self.covar = self.ratio*diff + (1.-self.ratio)*self.covar
        if value>self.avg+bound*self.covar:
            return True
        else:
            return False