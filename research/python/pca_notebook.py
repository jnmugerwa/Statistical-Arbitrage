"""

Originally a Jupyter Notebook, so this its messy :)

"""

# Period of decomposition.
start = "2018-10-01"
end = "2019-10-04"

# Construct a portfolio, whose returns we're decomposing.
portfolio = ['IBM', 'AAPL', 'FB', 'INTC', 'MSFT', 'DOVA', 'BVSN', 'AVYA', 'GLD']

# Get log returns of portfolio -- to be decomposed.
portfolio_returns = get_pricing(portfolio, start_date=start, end_date=end, fields="price").pct_change()[1: ]

# DECOMPOSEEEEEE
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Number of dims to keep.
num_pc = 2

# Converts list to np array.
X = np.asarray(portfolio_returns)

'''
cols = a time step for all m stocks
rows = a stock's return at all n timesteps
'''
[n, m] = X.shape

# PCA abstraction -- OOP style.
pca = PCA(n_components = num_pc)

'''
Fits the model with our data. Essentially computes the eigenvectors (here, the 
2 vectors on which we'll project our data to reduce dimensionality to 2 and 
keep major variation.)
'''
pca.fit(X)

'''
# The amount of variance from the selected eigenvectors / total variance from 
all eigenvectors. i.e. what % of the entire variance are these 2 vectors/dimensions?
'''
explained_var_ratio = pca.explained_variance_ratio_
cum_sum_of_first_k_eigens = np.cumsum(explained_var_ratio)
print ("{:0.2f}% of the return data's variance is explained by the first {} PC's."
       .format(cum_sum_of_first_k_eigens[-1]*100, 2))

x = np.arange(1, len(explained_var_ratio) + 1, 1)

plt.subplot(1, 2, 1)
plt.bar(x, explained_var_ratio*100, align = "center")
plt.title('Contribution of principal components',fontsize = 16)
plt.xlabel('principal components',fontsize = 16)
plt.ylabel('percentage',fontsize = 16)
plt.xticks(x,fontsize = 16) 
plt.yticks(fontsize = 16)
plt.xlim([0, num_pc+1])

plt.subplot(1, 2, 2)
plt.plot(x, cum_sum_of_first_k_eigens*100,'ro-')
plt.xlabel('principal components',fontsize = 16)
plt.ylabel('percentage',fontsize = 16)
plt.title('Cumulative contribution of principal components',fontsize = 16)
plt.xticks(x,fontsize = 16) 
plt.yticks(fontsize = 16)
plt.xlim([1, num_pc])
plt.ylim([50,100]);

# Array of principal axes.
pca_components = pca.components_

'''
Dot product of the portfolio returns and pca_components -- we're essentially 
seeing how codimensional the returns and factors principal components are. 
Factor returns are then:
'''
factor_returns = X.dot(pca_components.T)
factor_returns = pd.DataFrame(columns=["factor 1", "factor 2"], 
                              index=portfolio_returns.index,
                              data=factor_returns)
factor_returns.head()

factor_exposures = pd.DataFrame(index=["factor 1", "factor 2"], 
                                columns=portfolio_returns.columns,
                                data = pca.components_).T
factor_exposures

labels = factor_exposures.index
data = factor_exposures.values

plt.subplots_adjust(bottom = 0.1)
plt.scatter(
    data[:, 0], data[:, 1], marker='o', s=300, c='m',
    cmap=plt.get_cmap('Spectral'))
plt.title('Scatter Plot of Coefficients of PC1 and PC2')
plt.xlabel('factor exposure of PC1')
plt.ylabel('factor exposure of PC2')

for label, x, y in zip(labels, data[:, 0], data[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
    );