import theano as T
import theano.tensor as TT

    #return TT.nnet.sigmoid(X)

def Copula_i(u,v,teta=0):
    return u*v

def Copula_g(u,v,teta=1.2):
    U = (-TT.log(u))**teta
    V = (-TT.log(v))**teta
    return TT.exp(-((U+V)**(1/teta)))

def Copula_f(u,v,teta=-2):
    U = (TT.exp( -teta*u )-1)
    V = (TT.exp( -teta*v )-1)
    S = TT.exp( -teta ) - 1
    return -(1/teta) * TT.log( 1 + (U*V)/S )

