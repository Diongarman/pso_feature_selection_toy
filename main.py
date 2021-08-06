# Import modules
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pyswarms as ps
from sklearn import linear_model

from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score


from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)

X, y = make_classification(n_samples=100, n_features=15, n_classes=3,
                           n_informative=5, n_redundant=1, n_repeated=3,
                           random_state=1)

                           

# Plot toy dataset per feature
df = pd.DataFrame(X)
df['labels'] = pd.Series(y)



#sns.pairplot(df).fig.set_size_inches(10,10)

#plt.show()

# Create an instance of the classifier
classifier = linear_model.LogisticRegression()

# Define objective function

a=0
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
    classifier.fit(X_subset, y)
    #P = (classifier.predict(X_subset) == y).mean()
    P = (classifier.score(X_subset, y))

    #P = roc_auc_score(y, classifier.predict_proba(X_subset), multi_class='ovr')

    
    # Compute for the objective function --> returns error in data
    j = ((alpha * (1.0 - P))
        + (1.0 - alpha) * ((X_subset.shape[1] / total_features)))

    
    return j


def f(x, alpha=0.75):
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
    global a
    print(a)
    
    a=a+1
    
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]

    print(min(j))
    #print(j)

    return np.array(j)



# Initialize swarm, arbitrary: See academic papers on initialisations
options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 30, 'p':2}

# Call instance of PSO
dimensions = 15 # dimensions should be the number of features
#optimizer.reset()
optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options=options)

# Perform optimization
cost, pos = optimizer.optimize(f,  iters=100, verbose=True)


# Create two instances of LogisticRegression
c1 = linear_model.LogisticRegression()
c2 = linear_model.LogisticRegression()

# Get the selected features from the final positions
X_selected_features = X[:,pos==1]  # subset

# Perform classification and store performance in P
c1.fit(X_selected_features, y)
c2.fit(X, y)

# Compute performance
subset_performance = c1.score(X_selected_features, y)
allfeatures_performance = c2.score(X,y)


print('Subset performance: %.3f' % (subset_performance))
print('Superset performance: %.3f' %(allfeatures_performance))

df1 = pd.DataFrame(X_selected_features)
df1['labels'] = pd.Series(y)

sns.pairplot(df1, hue='labels').fig.set_size_inches(10,10)

plt.show()

plot_cost_history(cost_history=optimizer.mean_neighbor_history)
plt.show()


# Enables us to view it in a Jupyter notebook
#HTML(animation.to_html5_video())

optimizer.reset()