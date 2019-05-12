import numpy as np
import emcee
from scipy import stats
import scipy.optimize as op
import matplotlib.pyplot as plt

mu1 = 0.0
sigma1 = 1.0
 
mu2 = 4.0
sigma2= 3.0

N =100

sample1 = np.random.normal(mu1,sigma1,N)
sample2 = np.random.normal(mu2,sigma2,N)

print("sample1 std", np.std(sample1))
print("sample2 std", np.std(sample2))

#std deviation
var_a = sample1.var(ddof=1)
var_b = sample2.var(ddof=1)

s = np.sqrt((var_a + var_b)/2)


## Calculate the t-statistics
t = (sample1.mean() - sample2.mean())/(s*np.sqrt(2/N))



## Compare with the critical t-value
#Degrees of freedom
df = 2*N - 2

#p-value after comparison with the t 
p = 1 - stats.t.cdf(t,df=df)


print("t = " + str(t))
print("p = " + str(2*p))
### You can see that after comparing the t statistic with the critical t value (computed internally) we get a good p value of 0.0005 and thus we reject the null hypothesis and thus it proves that the mean of the two distributions are different and statistically significant.


## Cross Checking with the internal scipy function
t2, p2 = stats.ttest_ind(sample1,sample2)
print("t = " + str(t2))
print("p = " + str(p2))



def lnlike(theta,s1,s2):
	mu1,mu2,var1,var2 = theta
	
	ds1 = 0.0
	ds2 = 0.0
	ds = 0.0
	
	N1 = len(s1)
	N2 = len(s2)
	
	inv_var1 = (1.0/var1)
	inv_var2 = (1.0/var2)
	
	s1_x1 = np.mean(s1)
	s1_x2 = np.mean(s1*s1)
	
	s2_x1 = np.mean(s2)
	s2_x2 = np.mean(s2*s2)
	
	
	ds1 = -0.5*N1*(mu1**2-2.0*s1_x1*mu1+s1_x2)*inv_var1-0.5*N1*np.log(var1)
	ds2 = -0.5*N2*(mu2**2-2.0*s2_x1*mu2+s2_x2)*inv_var2-0.5*N2*np.log(var2)
	
		
	ds+= (ds1+ds2)
	
	#ds = np.exp(ds)
	
	return ds
	
def lnprior(theta):
	
	mu1,mu2,var1,var2 = theta
	
	if(-5.0<mu1<5.0 and -5.0<mu2<5.0 and 0.5 < var1 < 20.0 and 0.5 < var2 < 20.0):
		return 0.0
		#return -np.log(var1)-np.log(var2)
	
	
	return -np.inf
	
def lnprob(theta,s1,s2):
	
	lp = lnprior(theta)
	
	if not np.isfinite(lp):
		return -np.inf
	
	
	return lp+lnlike(theta,s1,s2)

ndim, nwalkers = 4, 200
pos = [[1.0,1.0,1.0,1.0] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(sample1, sample2))
sampler.run_mcmc(pos, 1000)
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

mu1_samples = samples[:,0]
mu2_samples = samples[:,1]

var1_samples = np.sqrt(samples[:,2])
var2_samples = np.sqrt(samples[:,3])

print("======================================")
print(np.mean(mu1_samples),np.std(mu1_samples))
print("======================================")
plt.hist(mu1_samples)
plt.show()

print("======================================")
print(np.mean(mu2_samples),np.std(mu2_samples))
print("======================================")
plt.clf()
plt.hist(mu2_samples)
plt.show()

print("======================================")
print(np.mean(var1_samples),np.std(var1_samples))
print("======================================")
plt.clf()
plt.title("Var1")
plt.hist(var1_samples)
plt.show()

print("======================================")
print(np.mean(var2_samples),np.std(var2_samples))
print("======================================")
plt.clf()
plt.title("Var2")
plt.hist(var2_samples)
plt.show()


print("======================================")
print(np.mean(var1_samples/var2_samples),np.std(var1_samples/var2_samples))
print("======================================")
plt.clf()
plt.title("Var1/Var2")
plt.hist(var1_samples/var2_samples)
plt.show()

