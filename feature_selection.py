import glob
# Import modules
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# Import PySwarms
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history
# sklearn imports
from sklearn import linear_model
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

class FSS_PSO_Builder:


    d = {'best cost': [], 'msle (subset)': [],'msle (all)':[], 'r2 (subset)':[], 'r2 (all)':[], 'mae (subset)':[], 'mae (all)':[],'ratio selected':[],'selected features': []}
    #data = pd.read_excel(file_name)
    # performance_metric -> pass in list of sklearn metrics later
    def __init__(self, options, data, n_particles,it, regressor ,performance_metric,alpha, obj_function_equation, final_eval_ML_model_1, final_eval_ML_model_2):
        index = len(data.columns)
        self.X = (data.iloc[:, 0:index-1]).to_numpy()
        self.y = (data.iloc[:, -1]).to_numpy()
        self.dimensions = self.X.shape[1]# dimensions should be the number of features
        self.columns = list(data.columns)
        self.regressor = regressor
        self.m1 = final_eval_ML_model_1
        self.m2 = final_eval_ML_model_2
        self.obj_function_equation = obj_function_equation
        self.optimizer = ps.discrete.BinaryPSO(n_particles=n_particles, dimensions=self.dimensions, options=options)# Call instance of PSO
        self.cost, self.pos = self.optimizer.optimize(self.f,  iters=it, verbose=True, X=self.X, y=self.y, performance_metric = performance_metric, alpha=alpha)# Perform optimization
    
    def f(self, swarm, X,y, performance_metric,alpha):
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

        j = [
            self.f_per_particle(swarm[particle], alpha, X, y, performance_metric) 
            for particle in range(n_particles)
            ]
        
        print(min(j))
        print(j.index(min(j)))
        return np.array(j)
    def f_per_particle(self,m, alpha, X, y, P):
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
        fitness_error = make_scorer(self.__objective_fcn,  ratio_selected_features=ratio_selected_features, P=P, alpha=alpha)
        scores = cross_val_score(self.regressor, X_subset, y, cv=10, scoring=fitness_error)
    
        j = scores.mean()
        return j

    def __objective_fcn(self, y_true, y_pred, **kwargs):

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
        #print(kwargs['P'])
        #kwargs['ratio_selected_features'] is objective 2
        
        j = self.obj_function_equation(p, kwargs['ratio_selected_features'], kwargs['alpha'])
        
        return j


    def do_cross_val(self):
        # Get the selected features from the final positions
        X_selected_features = self.X[:,self.pos==1]# subset


        # Compute R2 estimate using CV
        r2_subset = cross_validate(self.m1, X_selected_features, self.y, cv=10, scoring='r2')
        r2_all = cross_validate(self.m2, self.X, self.y, cv=10, scoring='r2')

        # Get mean R2 estimate obtained from CV
        self.r2_subset = r2_subset['test_score'].mean()
        self.r2_all = r2_all['test_score'].mean()
        
        # Compute msle estimate using CV
        msle_subset = cross_validate(self.m1, X_selected_features, self.y, cv=10, scoring='neg_mean_squared_log_error')
        msle_all = cross_validate(self.m2, self.X, self.y, cv=10, scoring='neg_mean_squared_log_error')

        # Get mean msle estimate obtained from CV
        self.msle_subset = msle_subset['test_score'].mean()
        self.msle_all = msle_all['test_score'].mean()

        # Compute mae estimate using CV
        mae_subset = cross_validate(self.m1, X_selected_features, self.y, cv=10, scoring='neg_mean_absolute_error')
        mae_all = cross_validate(self.m2, self.X, self.y, cv=10, scoring='neg_mean_absolute_error')

        # Get mean mae estimate obtained from CV
        self.mae_subset = mae_subset['test_score'].mean()
        self.mae_all = mae_all['test_score'].mean()

    def create_results(self):
        FSS_PSO_Builder.d['best cost'].append(self.cost)
        FSS_PSO_Builder.d['r2 (subset)'].append(self.r2_subset)
        FSS_PSO_Builder.d['r2 (all)'].append(self.r2_all)        
        FSS_PSO_Builder.d['msle (subset)'].append(self.msle_subset)
        FSS_PSO_Builder.d['msle (all)'].append(self.msle_all)
        FSS_PSO_Builder.d['mae (subset)'].append(self.mae_subset)
        FSS_PSO_Builder.d['mae (all)'].append(self.msle_all)       
        FSS_PSO_Builder.d['ratio selected'].append(self.selected_features_ratio)
        FSS_PSO_Builder.d['selected features'].append(self.__get_feature_col_names())#data is global - consider changing this

        
    def count_selected_features(self):
        self.num_selected = np.count_nonzero(self.pos)
        self.total_feats = len(self.pos)
        self.selected_features_ratio = "{selected}/{total}".format(selected = str(self.num_selected), total = str(self.total_feats))
    
    def __get_feature_col_names(self):

        """ 
        
        inputs
        -------

        outputs
        -------
        
        """
        
        indices = [i for i, x in enumerate(self.pos) if x == 1]
        cols = [self.columns[i] for i in indices]

        return cols
    
    def viz_cost_history(self):

        plot_cost_history(cost_history=self.optimizer.mean_neighbor_history)
        plt.show()

    def viz_scatter_plot_matrix(self):
        X_selected_features = self.X[:,self.pos==1]
        df1 = pd.DataFrame(X_selected_features)

        #df1['labels'] = pd.Series(y)

        sns.pairplot(df1,height=5, aspect=.8, kind="reg")
        plt.show()            
    
    def build(self):
        return FSS_PSO(self.columns, self.optimizer, self.cost,  self.pos, self.selected_features_ratio, FSS_PSO_Builder.d)
        
        #a.do_cross_val()
        #a.count_selected_features()    


class FSS_PSO:
    
    def __init__(self, columns, optimizer, cost, pos,  selected_features_ratio, results):
        #all initialisations from builder
        self.columns = columns
        self.optimizer = optimizer
        self.cost = cost
        self.pos = pos

        self.selected_features_ratio = selected_features_ratio
        self.results = results
        #new shit
    
    def save_results_csv(self):
        results = pd.DataFrame(data=self.results)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(results)

        results.to_csv('results.csv')

        return results

    def get_selected_feature_history(self):
        flat_feat_freq_list = self.__flatten(self.results['selected features'])
        return flat_feat_freq_list

    def __flatten(self,t):
        return [item for sublist in t for item in sublist]

    def plot_feat_frequency(self,):

        flat_list = self.get_selected_feature_history()

        feat_freq_dict = dict((x,flat_list.count(x)) for x in set(flat_list))
        plt.bar(feat_freq_dict.keys(), feat_freq_dict.values())
        plt.show()



#OF Expressions

def OF_equation(obj_1, obj_2, alpha):
    #print(obj_1)
    j = (alpha * (1-obj_1) + (1.0 - alpha) * (obj_2))
    return j

#GLOBALS
candidate_solutions = {}
#d = {'best cost': [], 'subset performance': [], 'fitness stedv': [],'full dataset performance': [], 'ratio selected':[],'selected features': []}
options = {'c1': 0.5, 'c2': 0.5, 'w':0.3, 'k': 30, 'p':2}
data = pd.read_excel('drivPoints.xlsx')
regressor = linear_model.LinearRegression()
m1 = linear_model.LinearRegression()
m2 = linear_model.LinearRegression()



def run(n):

    for x in range(n):

        #Mandatory function calls

        #partially initialise
        a = FSS_PSO_Builder(options, data, 30,10,regressor,r2, 0.5,OF_equation, m1,m2)
        #initialise other fields
        a.count_selected_features()
        a.do_cross_val()
        #add results
        a.create_results()

        #optional function calls
        a.viz_cost_history()
        a.viz_scatter_plot_matrix()

        #reset optimiser for next iteration
        a.optimizer.reset()
    return a


a = run(2)
#build fully initialised object
b = a.build()
#save results as csv
b.save_results_csv()
#save solution - later get code running first
#reset optimiser
b.optimizer.reset()
#visualisations
b.plot_feat_frequency()

