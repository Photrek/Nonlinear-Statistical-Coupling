import numpy as np
import matplotlib.pyplot as plt
import site
from importlib import reload
reload(site)
import sys #Changing the system path
sys.path.insert(0, '/home/hongxiang/Documents/repos/Nonlinear-Statistical-Coupling')

#Importing different versions of CoupledNormal
from nsc.distributions.coupled_normal import CoupledNormal
from nsc.distributions.coupled_normal_tf import CoupledNormal as CoupledNormal_tf

##Testing
# Define the loc, scale, alpha, and kappas that will be used for all distributions.
loc, scale = 0., 1.
alpha = 2
n_sample = 10000

# Define the values the various Coupled Gaussians can range from -6*scale to 6*scale
X = np.linspace(-6*scale, 6*scale, n_sample)

# Plot Coupled Gaussians (loc, scale) with alpha = 2, with kappas going from 0 to 1
# by steps of 0.1
fig, ax = plt.subplots(figsize=(14, 8))
ax.axvline(c='black', lw=1)
ax.axhline(c='black', lw=1)

# plt.figure(figsize=(12, 6), dpi=100, facecolor='w', edgecolor='k')

cm = plt.get_cmap('coolwarm')
kappa_values = np.linspace(0, 1, 21)
n = len(kappa_values)
ax.set_prop_cycle(color=[cm(1.*i/n) for i, kappa in enumerate(kappa_values)])

#Storing pdf values
cn_pdf_vals_true = []
cn_pdf_vals_tf = []
for kappa in kappa_values:
    #Original
    temp_normal = CoupledNormal(loc=loc,
                                    scale=scale,
                                    kappa=round(kappa, 2),
                                    alpha=alpha
                                    )
    #Tensorflow
    temp_normal_tf = CoupledNormal_tf(loc=loc,
                                    scale=scale,
                                    kappa=round(kappa, 2),
                                    alpha=alpha
                                    )
    plt.plot(X, temp_normal.prob(X), label=f'kappa = {temp_normal.kappa}')
    
    cn_pdf_vals_true.append(temp_normal.prob(X))
    cn_pdf_vals_tf.append(temp_normal_tf.prob(X))

plt.title(f'Coupled Normal ({loc}, {scale}) PDF with alpha = {alpha}')
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend()
plt.show();

cn_pdf_vals_true = np.array(cn_pdf_vals_true)
cn_pdf_vals_tf = np.array(cn_pdf_vals_tf)

print((cn_pdf_vals_true == cn_pdf_vals_true).all())

# Checking the sampling
kappa = 0.5
scale=10.0
sample_size = 1000
bins = 30

#CoupledNormal
cn = CoupledNormal(loc=loc, scale=scale, kappa=kappa)
cn_samples = cn.sample_n(sample_size)

#CoupledNormal_tf
cn_tf = CoupledNormal_tf(loc=loc, scale=scale, kappa=kappa)
cn_tf_samples = cn_tf.sample_n(sample_size)

#Scipy students_t
#s_t_samples = t.rvs(df=1/kappa, loc=loc, scale=scale, size=sample_size)

fig, axs = plt.subplots(3, 1, figsize=(12, 16))
#plt.setp(axs, xlim=(-10*scale,10*scale))
axs[0].hist(cn_samples, bins=bins, label='CoupledNormal')
axs[1].hist(cn_tf_samples, bins=bins, label='CoupledNormal_tf')
#axs[2].hist(s_t_samples, bins=bins, label='Students-t')
