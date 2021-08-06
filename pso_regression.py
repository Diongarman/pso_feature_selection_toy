# Import modules
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# Import PySwarms
import pyswarms as ps
from sklearn import linear_model
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.metrics import mean_squared_error as mse


from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=15, bias=0.3,
                           n_informative=5,
                           effective_rank=4,
                           random_state=1)


# Plot toy dataset per feature
df = pd.DataFrame(X)
df['labels'] = y

#sns.pairplot(df,height=5, aspect=.8, kind="reg")

#plt.show()



regressor = linear_model.LinearRegression()

def custom_fitness(y_true, y_pred):
    P = mse(y_true,y_pred)
    j = (alpha * (P) + (1.0 - alpha) * ((X_subset.shape[1] / total_features)))
    return j

def f_per_particle(m, alpha):
    """Computes for the objective function per particle

    Inputs
    ------
    m : numpy.ndarray
        Binary mask that can be obtained from BinaryPSO, will
        be used to mask features.
    alpha: float (default is 0.5)
        Constant weight for trading-off classifier performance
        and number of features

    Returns
    -------
    numpy.ndarray
        Computed objective function
    """
    total_features = 15
    # Get the subset of the features from the binary mask
    if np.count_nonzero(m) == 0:
        X_subset = X
    else:
        X_subset = X[:,m==1]
    # Perform classification and store performance in P
    regressor.fit(X_subset, y)
    P = mse(y,(regressor.predict(X_subset)))#breaking the golden rule!
    
    # Compute for the objective function
    #j = (alpha * (1.0 - P) + (1.0 - alpha) * ((X_subset.shape[1] / total_features)))
    j = (alpha * (P) + (1.0 - alpha) * ((X_subset.shape[1] / total_features)))
    return j
def f(x, alpha=0.5):
    """Higher-level method to do classification in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    
    n_particles = x.shape[0]

    #print('eyyyy',x)
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    print(min(j))
    return np.array(j)



# Initialize swarm, arbitrary: See academic papers on initialisations
options = {'c1': 0.5, 'c2': 0.5, 'w':0.3, 'k': 30, 'p':2}

# Call instance of PSO
dimensions = 15 # dimensions should be the number of features
#optimizer.reset()
optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options=options)

# Perform optimization
cost, pos = optimizer.optimize(f,  iters=100, verbose=True)
optimizer.reset()


# Create two instances of LinearRegression
c1 = linear_model.LinearRegression()

# Get the selected features from the final positions
X_selected_features = X[:,pos==1]  # subset

# Perform classification and store performance in P
c1.fit(X_selected_features, y)

# Compute performance
subset_performance = c1.score(X_selected_features, y)

print('Subset performance: %.3f' % (subset_performance))

df1 = pd.DataFrame(X_selected_features)
df1['labels'] = pd.Series(y)

sns.pairplot(df1,
             height=5, aspect=.8, kind="reg")

plt.show()