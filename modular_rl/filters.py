from .running_stat import RunningStat
import numpy as np

class Composition(object):
    def __init__(self, fs):
        self.fs = fs
    def __call__(self, x, update=True):
        for f in self.fs:
            x = f(x)
        return x
    def output_shape(self, input_space):
        out = input_space.shape
        for f in self.fs:
            out = f.output_shape(out)
        return out

class ZFilter(object):
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std+1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape

class ZFilterSelect(object):
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, indices=None, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip
        self.indices = indices

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x[self.indices])
        if self.demean:
            x[self.indices] = x[self.indices] - self.rs.mean
        if self.destd:
            x[self.indices] = x[self.indices] / (self.rs.std+1e-8)
        if self.clip:
            x[self.indices] = np.clip(x[self.indices], -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape

class Flatten(object):
    def __call__(self, x, update=True):
        return x.ravel()
    def output_shape(self, input_space):
        return (int(np.prod(input_space.shape)),)

class Ind2OneHot(object):
    def __init__(self, n):
        self.n = n
    def __call__(self, x, update=True):
        out = np.zeros(self.n)
        out[x] = 1
        return out
    def output_shape(self, input_space):
        return (input_space.n,)

class ConcatPrevious(object):
    def __init__(self, obspace):
        print(obspace.shape)
        self.prev = np.zeros(obspace.shape)

    def __call__(self, x, update=True):
        x = np.array(x)
        xc = np.concatenate([x, (x-self.prev)/0.01])
        self.prev = x
        return xc

    def output_shape(self, input_space):
        return ((input_space.shape[0]*2, ))

class FeatureInducer(object):
    def __init__(self, obspace):
        print(obspace.shape)
        self.prev = None #np.zeros(obspace.shape)
        self.concat_shape = 73
        self.filter_shape = self.concat_shape + 12
        self.final_shape = self.concat_shape + 12 + 2
        indices = np.delete(np.array(range(self.filter_shape)), [32, 33, 34, 35, 36])   #Do not Z-Filter Posas+Obstacle
        self.zfilter = ZFilterSelect(len(indices), indices, clip=5)

    def concatprevious(self, x):
        if self.prev is None:
            self.prev = np.zeros(x.shape)
        xdiff = (x-self.prev[:len(x)])/0.01
        xc = np.concatenate([x, xdiff[:-5]])
        return xc

    def addbodypartacc(self, x):
        xbpv = x
        if len(x) > len(self.prev):
            for i in range(12):
                xbpv = np.append(xbpv, 0)
        else:
            for i in range(self.concat_shape-12, self.concat_shape):         #Body parts accleration
                acc = (x[i]-self.prev[i])/0.01
                xbpv = np.append(xbpv, acc)
        return xbpv

    def modifyobstacle(self, x):
        obsrel = x[34]
        x[34] = min(4, max(-3, obsrel))/3   #Modifying obstacle distance
        falloff = min(1,max(0,3-abs(obsrel)))
        x = np.append(x, falloff*x[35])     #Adding falloff*obstacle height
        x = np.append(x, falloff*x[36])     #Adding falloff*obstacle radius 
        return x

    def __call__(self, x, update=True):
        x = np.array(x)
        px, py = x[1], x[2]
        x[18] -= px                     #COM_x relative to Pelvis_X
        x[19] -= py                     #COM_y relative to Pelvis_Y
        x = np.append(x, x[20]-x[4])    #COM_Velx relative to Pelvis_Velx
        x = np.append(x, x[21]-x[5])    #COM_Velx relative to Pelvis_Velx
        for i in range(22, 36):         #Body parts relative to Pelvis
            if i%2==0:
                x[i] -= px
            else:
                x[i] -= py
        x = np.delete(x, [1,2, 24, 25])# , 45, 46])
        x = self.concatprevious(x)
        x = self.addbodypartacc(x)
        x = self.zfilter(x)
        x = self.modifyobstacle(x)
        self.prev = x
        return x

    def output_shape(self, input_space):
        return ((self.final_shape, ))
