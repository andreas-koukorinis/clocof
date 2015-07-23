import numpy as np
import matplotlib.pyplot as plt
import theano as T
import theano.tensor as TT
from fun2d import fun2d
from copulas import *

def normcdf(X, nu = 0, sigma=1):
    return 0.5 * (1 + TT.erf( (X-nu) / (sigma * 2**0.5)))

## Generate correlated data
mean = [0,0]
cov = [[1,0.6],[0.6,1]]
x,y = np.random.multivariate_normal(mean,cov,1000).T

x = T.shared(x)
y = T.shared(y)

u = normcdf(x)
v = normcdf(y)

u_ = TT.dscalar('u')
v_ = TT.dscalar('v')
t_ = TT.dscalar('teta')

C_g = T.function(
    [u_,v_,t_],
    Copula_g(u_,v_,t_)
)

c_g = T.function(
    [u_,v_,t_],
    TT.log(T.grad(T.grad(Copula_g(u_,v_,t_),u_),v_))
)

print C_g(0.4,0.4,2)
print c_g(0.4,0.4,2)



ax1 = plt.subplot2grid((5,5), (0,0))
plt.axis('equal')
plt.plot(x.eval(),y.eval(),'x')
plt.title('A) joint distribution')
plt.xlabel('$z_1$')
plt.ylabel('$z_2$')

ax1 = plt.subplot2grid((5,5), (0,1))
plt.plot(u.eval(),v.eval(),'x')
plt.axis('equal')
plt.title('B) CDF')
plt.xlabel('$\Phi(z_1)$')
plt.ylabel('$\Phi(z_2)$')

for i in range(2,5):
    ax1 = plt.subplot2grid((5,5), (0,i))
    def f(x,y):return C_g(x,y,i)
    fun2d(f)
    plt.colorbar()
    plt.title('D) Copula')
    plt.xlabel('$\Phi(z_1)$')
    plt.ylabel('$\Phi(z_2)$')

teta = 0
for i in range(1,5):
    for j in range(5):
        ax1 = plt.subplot2grid((5,5), (i,j))
        teta+=1
        def f(x,y):return c_g(x,y,teta)
        fun2d(f)
    plt.xlabel('$\Phi(z_1)$')
    plt.ylabel('$\Phi(z_2)$')
    plt.title('C) Copula Density')
plt.colorbar()

#plt.tight_layout()
plt.savefig('/tmp/tmp.pdf')
