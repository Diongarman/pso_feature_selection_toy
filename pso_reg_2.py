import glob


# Import modules
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# Import PySwarms
import pyswarms as ps
from sklearn import linear_model
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2

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

print(X.shape)

# Plot toy dataset per feature
df = pd.DataFrame(X)
df['labels'] = y

#sns.pairplot(df,height=5, aspect=.8, kind="reg").fig.set_size_inches(10,10)

#plt.show()

classifier = linear_model.LinearRegression()

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
    total_features = X.shape[1]
    # Get the subset of the features from the binary mask
    if np.count_nonzero(m) == 0:
        X_subset = X
    else:
        X_subset = X[:,m==1]
    # Perform classification and store performance in P
    classifier.fit(X_subset, y)
    #P = -mse(y,(classifier.predict(X_subset)))
    #P = (classifier.score(X_subset, y))

    #P = roc_auc_score(y, classifier.predict_proba(X_subset), multi_class='ovr')

    
    # Compute for the objective function
    j = (alpha * (1.0 - P)
        + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))

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
    return np.array(j)



# Initialize swarm, arbitrary: See academic papers on initialisations
options = {'c1': 0.5, 'c2': 0.5, 'w':0.3, 'k': 30, 'p':2}

# Call instance of PSO
dimensions = X.shape[1] # dimensions should be the number of features
#optimizer.reset()
optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options=options)

# Perform optimization
cost, pos = optimizer.optimize(f,  iters=100, verbose=True)
optimizer.reset()



print(pos)
# Create two instances of LinearRegression
c1 = linear_model.LinearRegression()

# Get the selected features from the final positions
X_selected_features = X[:,pos==1]  # subset

# Perform classification and store performance in P
c1.fit(X_selected_features, y)

predicted_y_values = c1.predict(X_selected_feature)
r2_score_final = r2(y, predicted_y_values)

# Compute performance
#subset_performance = c1.score(X_selected_features, y)

#print('Subset performance: %.3f' % (subset_performance))

#df1 = pd.DataFrame(X_selected_features)
#df1['labels'] = pd.Series(y)

#sns.pairplot(df1,height=5, aspect=.8, kind="reg")

#plt.show()

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
