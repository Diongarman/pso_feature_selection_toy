# Import modules
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#pyswarms imports
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history

#sklearn imports
from sklearn import linear_model
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score, accuracy_score as accuracy, make_scorer
from sklearn.model_selection import cross_val_score, cross_validate


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

def objective_fcn(y_true, y_pred, **kwargs):

    """ Higher-level function to compue the objective function value for a particle 
    
    Inputs
    ------

    y_true:

    y_pred:

    **kwargs: arguments needed to compute objective function. this will be dependent upon the function expression.
    
    Will be things like: 
        score/loss, 
        total_number_of_features, 
        number_of_selected_features, 
        alpha: The balancing value
    
    """
    p = kwargs['P'](y_true,y_pred) #objective 1
    #kwargs['ratio_selected_features'] is objective 2
    
    j = obj_function_equation(p, kwargs['ratio_selected_features'], kwargs['alpha'])
    
    return j

def obj_function_equation(obj_1, obj_2, alpha):
    j = (alpha * (1-obj_1) + (1.0 - alpha) * (obj_2))
    return j

def f_per_particle(m, alpha, X, y, P, ML_Algo):
    """Computes for the objective function per particle

    Inputs
    ------
    m : numpy.ndarray
        Binary mask that can be obtained from BinaryPSO, will
        be used to mask features.
    alpha: float (default is 0.5)
        Constant weight for trading-off classifier/regressor performance
        and number of features
    X: data to be used for CV
    y: labels to be used for CV

    Returns
    -------
    numpy.ndarray
        Computed objective function
    """
    total_features = X.shape[1]
    # Get the subset of the features from the binary mask
    if np.count_nonzero(m) == 0:
        X_subset = X
    else:
        X_subset = X[:,m==1]

    ratio_selected_features = X_subset.shape[1]/total_features
    
    #Particle fittness error/loss computed using cross validation
    fitness_error = make_scorer(objective_fcn,  ratio_selected_features=ratio_selected_features, P=P, alpha=alpha)
    scores = cross_val_score(ML_Algo, X_subset, y, cv=10, scoring=fitness_error)
  
    j = scores.mean()
    return j
def f(swarm, X,y, performance_metric,alpha, ML_Algo):
    """Higher-level method to do classification/regression in the
    whole swarm.

    Inputs
    ------
    swarm: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    X: data to pass into f_per_particle() function
    
    y: label data to pass into f_per_particle() function

    alpha: 

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    
    n_particles = swarm.shape[0]

    j = [f_per_particle(swarm[particle], alpha, X, y, performance_metric, ML_Algo) for particle in range(n_particles)]
    return np.array(j)


                                        ###############
                                        # Driver Code #
                                        ###############

# Initialize swarm, arbitrary: See academic papers on initialisations
options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 30, 'p':2}

# Call instance of PSO
dimensions = X.shape[1] # dimensions should be the number of features
#optimizer.reset()
optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options=options)

# Perform optimization
cost, pos = optimizer.optimize(f,  iters=100, verbose=True, X=X, y=y, performance_metric = accuracy, ML_Algo = classifier,alpha=0.75)

# Create two instances of LogisticRegression
c1 = linear_model.LogisticRegression()
c2 = linear_model.LogisticRegression()

# Get the selected features from the final positions
X_selected_features = X[:,pos==1]  # subset

# Compute performance using CV
scores = cross_validate(c1, X_selected_features, y, cv=10, scoring='accuracy')
scores2 = cross_validate(c2, X, y, cv=10, scoring='accuracy')

subset_performance = scores['test_score'].mean()
wholeset_performance = scores2['test_score'].mean()


print('Subset fitness cost/loss: %.3f' % (cost))
print('Subset performance: %.3f' % (subset_performance))
print('Full set performance: %.3f' % (wholeset_performance))

#get feature-column names
def get_feature_col_names(data, bit_list):

    """ 
    
    inputs
    -------

    outputs
    -------
    
    """
    cols = list(data.columns)
    indices = [i for i, x in enumerate(bit_list) if x == 1]
    cols = [cols[i] for i in indices]

    return cols

cols = get_feature_col_names(df, pos)
print('best columns', cols)

df1 = pd.DataFrame(X_selected_features)
df1['labels'] = pd.Series(y)

sns.pairplot(df1, hue='labels').fig.set_size_inches(10,10)

plt.show()

plot_cost_history(cost_history=optimizer.mean_neighbor_history)
plt.show()

optimizer.reset()