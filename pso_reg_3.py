import glob


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
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import cross_val_score, cross_validate

for file_name in glob.glob('*.xlsx'):

    print('Processing ' + file_name + '...')
    try:
        data = pd.read_excel(file_name)
        y = (data.iloc[:, -1]).to_numpy()
        index = len(data.columns)
        X = (data.iloc[:, 0:index-1]).to_numpy()

        file_name = file_name.split('.')[0]
    except:
        print('Error loading file ', file_name)


# Plot toy dataset per feature
df = pd.DataFrame(X)
df['labels'] = y

#sns.pairplot(df,height=5, aspect=.8, kind="reg").fig.set_size_inches(10,10)

#plt.show()

regressor = linear_model.LinearRegression()
#regressor = RandomForestRegressor(max_depth=2)

#measure fitness error - optimiser aims to find minima solution
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
    P = r2(y_true,y_pred)
    j = obj_function_equation(P, kwargs['total_feats'],kwargs['num_selected_features'], kwargs['alpha'])
    
    return j

def obj_function_equation(P, tot_features, num_selected_features,alpha):
    j = (alpha * (1-P) + (1.0 - alpha) * ((num_selected_features/tot_features)))
    return j

def f_per_particle(m, alpha, X, y):
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

    #Particle fittness error/loss computed using cross validation
        #greater_is_better=False, -> argument is removed from make_scorer
    fitness_error = make_scorer(objective_fcn,  num_selected_features=X_subset.shape[1], total_feats=total_features, alpha=alpha)
    scores = cross_val_score(regressor, X_subset, y, cv=10, scoring=fitness_error)
  
    j = scores.mean()
    return j
def f(swarm, X,y, alpha=0.5):
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

    j = [f_per_particle(swarm[particle], alpha, X, y) for particle in range(n_particles)]
    return np.array(j)

                                        ###############
                                        # Driver Code #
                                        ###############

# Initialize swarm, arbitrary: See academic papers on initialisations
options = {'c1': 0.5, 'c2': 0.5, 'w':0.3, 'k': 30, 'p':2}

# Call instance of PSO
dimensions = X.shape[1] # dimensions should be the number of features
#optimizer.reset()
optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options=options)

# Perform optimization
cost, pos = optimizer.optimize(f,  iters=100, verbose=True, X=X, y=y)
optimizer.reset()


# Create two instances of LinearRegression
r1 = linear_model.LinearRegression()
r2 = linear_model.LinearRegression()

# Get the selected features from the final positions
X_selected_features = X[:,pos==1]  # subset

# Compute performance using CV
scores = cross_validate(r1, X_selected_features, y, cv=10, scoring='r2')
scores2 = cross_validate(r2, X, y, cv=10, scoring='r2')

subset_performance = scores['test_score'].mean()
wholeset_performance = scores2['test_score'].mean()

print('Subset performance: %.3f' % (subset_performance))
print('Full set performance: %.3f' % (wholeset_performance))


# Compute performance
#subset_performance = c1.score(X_selected_features, y)

#print('Subset performance: %.3f' % (subset_performance))

df1 = pd.DataFrame(X_selected_features)
#df1['labels'] = pd.Series(y)

sns.pairplot(df1,height=5, aspect=.8, kind="reg")

plt.show()

#Objective function
#negative error
#check score function mechanics

#compare performances with FSS and none

# r2 training subset
# X_r2_train_selected_features = X_train[:, pos_r2_train == 1]
# r2 testing subset
# X_r2_test_selected_features = X_test[:, pos_r2_train == 1]
# regression_model.fit(X_r2_train_selected_features, y_train)
# predicted_values_r2_train = regression_model.predict(X_r2_train_selected_features)
# predicted_values_r2_test = regression_model.predict(X_r2_test_selected_features)
