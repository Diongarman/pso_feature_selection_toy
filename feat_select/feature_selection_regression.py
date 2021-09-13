# Import modules
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# Import PySwarms
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history
from pyswarms_monkey_patch.pso_parameter_tuning import RandomSearchUpdate, GridSearchUpdate
# sklearn imports
# learning algos
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
# metrics
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
#stats
from scipy.stats import pearsonr
# CV
from sklearn.model_selection import cross_val_score, cross_validate
#time related
from codetiming import Timer
import time
#Native
import os


matplotlib.use('Agg') #stops plots being displayed

class Repeated_Experiment_Results:
    
    def __init__(self, 
    columns, 
    optimizer, 
    cost, 
    pos,  
    selected_features_ratio, 
    results, 
    cost_histories, config_id):
        #all initialisations from builder
        self.columns = columns
        self.optimizer = optimizer
        self.cost = cost
        self.pos = pos

        self.selected_features_ratio = selected_features_ratio
        self.results = results
        self.cost_histories = cost_histories
        #new shit
        self.config_uid = config_id 
    
    def save_results_csv(self):
        results = pd.DataFrame(data=self.results)

        #show results
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(results)
        
        #Save results
        results.to_csv('Results_CSV/results.csv')

        return results

    def __get_selected_feature_history(self):
        flat_feat_freq_list = self.__flatten(self.results['selected features'])
        return flat_feat_freq_list

    def __flatten(self,t):
        return [item for sublist in t for item in sublist]


    #Todo: if too many iterations then plot IQR error band like plot: https://stackoverflow.com/questions/61888674/can-you-plot-interquartile-range-as-the-error-band-on-a-seaborn-lineplot
    def __cost_histories_box_plots(self):

        df = pd.DataFrame(self.cost_histories)
        
        avg_line = df.mean().to_frame()
        avg_line.columns = ['Avg']
        #print(df)
        #print(avg_line)
        ax = sns.boxplot(data=df)
        sns.swarmplot(data=df)#Todo: color by run i.e. rowise

        sns.lineplot(data=avg_line)

        ax.set(xlabel="Swarm Iteration", ylabel="Fitness Cost/Error")
        plt.show()

    def __plot_feat_frequency(self):

        flat_list = self.__get_selected_feature_history()

        feat_freq_dict = dict((x,flat_list.count(x)) for x in set(flat_list))

        fig = plt.figure()
        bar = plt.bar(feat_freq_dict.keys(), feat_freq_dict.values())

        

        

        file_name = "Experiment_Results/Viz/Aggregated/feature__frequency_{}".format(str(self.config_uid))

        fig.savefig(file_name, dpi=fig.dpi)
        plt.clf()

        #hist.savefig(file_name, dpi=400)

    
    def __plot_subset_size_hist(self):
        results = pd.DataFrame(data=self.results)
        print(results["ratio selected"])

        histo = sns.histplot(data=results, x="ratio selected", stat="count", discrete=True)
        
        file_name = "Experiment_Results/Viz/Aggregated/subset_size_{}".format(str(self.config_uid))

        hist_fig = histo.figure

        hist_fig.savefig(file_name)
        plt.clf()

    def aggregate_optional_functions(self, config):
        if config['plot_subset_size_histo']:
            self.__plot_subset_size_hist()
        if config['plot_feature_frequency']:
            self.__plot_feat_frequency()
        self.__cost_histories_box_plots()


class FSS_PSO_Experimental_Data_Collector:
    #results data aggregated over multiple runs
    cost_histories = []
    d = {'best fitness error': [], 'best fitness stdv':[], 'r2 (subset)':[], 'r2 (all)':[], 'mae (subset)':[], 'mae (all)':[],'ratio selected':[],'selected features': [], 'time':[]}
    #data = pd.read_excel(file_name)
    # performance_metric -> pass in list of sklearn metrics later
    def __init__(self, 
    data, 
    options, 
    n_particles,
    it, 
    regressor ,
    performance_metric,
    alpha, 
    obj_function_equation, 
    final_eval_ML_model_1, 
    final_eval_ML_model_2, 
    pso_parameter_optimiser, 
    config_id
    ):
        self.temp_best_cost = None
        self.cost_stdv = None
        
        self.columns = list(data.columns)
        self.X, self.y = self.__import_data(data)

        self.dimensions = self.X.shape[1]# dimensions = number of features

        self.regressor = regressor #drives fitness function
        #Post optimsiation evaluation metrics
        self.m1 = final_eval_ML_model_1
        self.m2 = final_eval_ML_model_2

        self.obj_function_equation = obj_function_equation
        #DEPENDENCY
        if pso_parameter_optimiser == 'random_search':
            bounds = self.__get_random_search_bounds()
            options = self.__random_search_parameter_optimiser(bounds, n_particles, self.dimensions,self.f, it, 2, self.X, self.y, performance_metric, alpha )
        #DEPENDENCY
        if pso_parameter_optimiser == 'grid_search':
            parameter_grid_seed = self.__get_grid_search_grid_seed()
            options = self.__grid_search_parameter_optimiser(parameter_grid_seed, n_particles, self.dimensions,self.f, it, self.X, self.y, performance_metric, alpha )

        #IDEA: maybe embedd timer into CV fitness code to obtain and be able to report avg train and test times
        t = Timer(name="class")
        t.start()
        #DEPENDENCY
        self.optimizer = ps.discrete.BinaryPSO(n_particles=n_particles, dimensions=self.dimensions, options=options)# Call instance of PSO
        self.cost, self.pos = self.optimizer.optimize(self.f,  iters=it, verbose=True, X=self.X, y=self.y, performance_metric = performance_metric, alpha=alpha)# Perform optimization
        self.optimisation_time = t.stop()

        self.config_id = config_id
        

    def __random_search_parameter_optimiser(self, opts, n_particles, dimensions, OF, it, n_selection_iters, X, y,  performance_metric, alpha):

        r = RandomSearchUpdate(
            ps.discrete.BinaryPSO, 
            n_particles=n_particles, 
            dimensions=dimensions,
            options=opts, 
            objective_func=OF, 
            iters=it, 
            n_selection_iters = n_selection_iters,
            X=X, 
            y=y, 
            performance_metric = performance_metric, 
            alpha=alpha)

        _ , best_options= r.search()

        return best_options
    
    def __grid_search_parameter_optimiser(self, opts, n_particles, dimensions, OF, it,  X, y,  performance_metric, alpha):

        g = GridSearchUpdate(
            ps.discrete.BinaryPSO, 
            n_particles=n_particles, 
            dimensions=dimensions,
            options=opts, 
            objective_func=OF, 
            iters=it, 
            X=X, 
            y=y, 
            performance_metric = performance_metric, 
            alpha=alpha)

        _ , best_options= g.search()

        return best_options

    #Todo: update this function to include 1)the ability to accept manual input from user 2) choose from educated values according to literature
    def __get_random_search_bounds(self):
        return {'c1': [.25, .75],
               'c2': [.25, .75],
               'w' : [.1, 1],
               'k' : [5, 50],#Todo: COmpute dynamically as a function of n_particle
               'p' : 1}
    #Todo: update this function to include 1)the ability to accept manual input from user 2) choose from educated values according to literature
    def __get_grid_search_grid_seed(self):
        return {'c1': [.25, .5, .75],
                        'c2': [.25, .5, .75],
                        'w' : [.25, .5, .75],
                        'k' : [5, 10, 15], #Todo: COmpute dynamically as a function of n_particles
                        'p' : 1}        

    def __import_data(self, data):
        index = len(self.columns)
        X = (data.iloc[:, 0:index-1]).to_numpy()
        y = (data.iloc[:, -1]).to_numpy()
        return X, y

    def __set_best_fitness_err_stdv(self, fitness_err_mean, fitness_err_stdv):

        #find minimum fitness error and it's associated stdev
        min_fitness_err = min(fitness_err_mean)
        mindex = fitness_err_mean.index(min_fitness_err)
        min_fitness_err_stdv = fitness_err_stdv[mindex]

        #Store record best (minimum) cost and it's stdev 
        if ((self.temp_best_cost is None) or (min_fitness_err < self.temp_best_cost)):
            self.temp_best_cost = min_fitness_err
            self.cost_stdv = min_fitness_err_stdv 

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

        #document shape of this data structure j = [(mean_1, stdev_1),....,(mean_it, stdev_it)]
        j = [
            self.f_per_particle(swarm[particle], alpha, X, y, performance_metric) 
                for particle in range(n_particles)
            ]
        fitness_error_mean, fitness_error_stdv = zip(*j)

        self.__set_best_fitness_err_stdv(fitness_error_mean, fitness_error_stdv)

        return np.array(fitness_error_mean)

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
        scores = cross_val_score(self.regressor, X_subset, y, cv=10, scoring=fitness_error) #does first 4 steps of k-fold CV
        particle_fitness_err_mean = scores.mean()

        #Stdev 
        particle_fitness_err_stdev = np.std(scores)


        return (particle_fitness_err_mean, particle_fitness_err_stdev)

    def __objective_fcn(self, y_true, y_pred, **kwargs):

        """ Higher-level function to compute the objective function value for a particle 
        
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
        obj1 = kwargs['P'](y_true,y_pred) #objective 1
        obj2 = kwargs['ratio_selected_features'] #is objective 2
        
        particle_value = self.obj_function_equation(obj1,obj2, kwargs['alpha'])
        
        return particle_value

    #Step 4 in generic wrapper framework - could be abstract method at some point that switches for validation in fuzzy framework, or could have two validation procedure. This one validates models accuracy and fuzzy does interpretability
    #Repurpose this function so that 
    def wrapper_validation_cross_val(self):
        # Get the selected features from the final positions
        X_selected_features = self.X[:,self.pos==1]# subset


        # Compute R2 estimate using CV
        r2_subset = cross_validate(self.m1, X_selected_features, self.y, cv=10, scoring='r2',return_train_score=True)
        r2_all = cross_validate(self.m2, self.X, self.y, cv=10, scoring='r2',return_train_score=True)

        # Get mean R2 estimate obtained from CV
        self.r2_subset = r2_subset['test_score'].mean()
        self.r2_all = r2_all['test_score'].mean()

        print(r2_subset['test_score'])
        print(r2_subset['train_score'])
        print(r2_all['test_score'])
        print(r2_all['train_score'])

        print(r2_subset['fit_time'])
        print(r2_subset['score_time'])
        print(r2_all['fit_time'])
        print(r2_all['score_time'])


        self.__viz_overfitting_cross_val(r2_subset['test_score'], r2_subset['train_score'], r2_all['test_score'], r2_all['train_score'])

        

        # Compute mae estimate using CV
        mae_subset = cross_validate(self.m1, X_selected_features, self.y, cv=10, scoring='neg_mean_absolute_error',return_train_score=True)
        mae_all = cross_validate(self.m2, self.X, self.y, cv=10, scoring='neg_mean_absolute_error',return_train_score=True)

        # Get mean mae estimate obtained from CV
        self.mae_subset = mae_subset['test_score'].mean()
        self.mae_all = mae_all['test_score'].mean()

    def __viz_overfitting_cross_val(self,r2_test_fss, r2_train_fss, r2_test, r2_train):
        folds = range(1, 10 + 1)
        fig = plt.figure()
        plt.plot(folds, r2_train_fss, 'o-', color='green', label='train subset')
        plt.plot(folds, r2_test_fss, 'o-', color='red', label='test subset')
        plt.plot(folds, r2_train, 'o-', color='blue', label='train')
        plt.plot(folds, r2_test, 'o-', color='orange', label='test')
        plt.legend()
        plt.grid()
        plt.xlabel('Number of fold')
        plt.ylabel('R2')
        #train_test.show()


        file_name = "Experiment_Results/Viz/Overfitting/overfitting_{}".format(str(self.config_id ))
        
        #train_test_fig = plt.figure

        
        fig.savefig(file_name, dpi=fig.dpi)
        plt.clf()
    def wrapper_validation_cross_val_TEMP(self):
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=10)
        mae_train = []
        mae_test = []
        for train_index, test_index in kf.split(self.X):
            
            X_train, X_test = self.X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = self.y[train_index], y[test_index]
           
            self.m1.fit(X_train, y_train)
            y_train_pred = self.m1.predict(X_train)
            y_test_pred = self.m1.predict(X_test)
            mae_train.append(mean_absolute_error(y_train, y_train_pred))
            mae_test.append(mean_absolute_error(y_test, y_test_pred))

    def create_results(self):
        FSS_PSO_Experimental_Data_Collector.d['best fitness error'].append(self.cost)
        FSS_PSO_Experimental_Data_Collector.d['best fitness stdv'].append(self.cost_stdv)
        FSS_PSO_Experimental_Data_Collector.d['r2 (subset)'].append(self.r2_subset)
        FSS_PSO_Experimental_Data_Collector.d['r2 (all)'].append(self.r2_all)        
        FSS_PSO_Experimental_Data_Collector.d['mae (subset)'].append(self.mae_subset)
        FSS_PSO_Experimental_Data_Collector.d['mae (all)'].append(self.mae_all)       
        #FSS_PSO_Experimental_Data_Collector.d['ratio selected'].append(self.selected_features_ratio)
        FSS_PSO_Experimental_Data_Collector.d['ratio selected'].append(self.num_selected)
        FSS_PSO_Experimental_Data_Collector.d['selected features'].append(self.__get_feature_col_names())#data is global - consider changing this
        FSS_PSO_Experimental_Data_Collector.d['time'].append(self.optimisation_time)

    def save_cost_history(self):
        #eveytime class is initialised and thus optimiser is run, store the cost history for that iteration in the class variable 'cost_histories'
        FSS_PSO_Experimental_Data_Collector.cost_histories.append(self.optimizer.cost_history)
     
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

    def __create_empty_folder(self, partial_path):
        curr_working_dir = os.path.abspath(__file__)
        
        curr_unix = int(time.time())
        newpath = curr_working_dir + partial_path + str(curr_unix) 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
    
    def viz_cost_history(self, experiment_n):

        line_plot = plot_cost_history(cost_history=self.optimizer.cost_history)
        #print('mean neighbour history', self.optimizer.mean_neighbor_history)
        #print(self.optimizer.cost_history)
        
        file_name = "Experiment_Results/Viz/Cost_History/cost_history_experiment_{}_{}".format(experiment_n, str(self.config_id ))
        
        line_plot_fig = line_plot.figure

        line_plot_fig.savefig(file_name)
        plt.clf()
    #Todo: Research stats videos and include interpretation of this viz in dissertation
    def viz_scatter_plot_matrix(self, experiment_n):
        
        X_selected_features = self.X[:,self.pos==1]
        df1 = pd.DataFrame(X_selected_features)

        df1.columns = self.__get_feature_col_names()

        #df1['labels'] = pd.Series(y)

        pair_plot = sns.pairplot(df1,height=5, aspect=.8, kind="reg",corner=True)

        pair_plot.map_lower(corrfunc)

        file_name = "Experiment_Results/Viz/Scatter/scatter_matrix_experiment_{}_{}".format(experiment_n, str(self.config_id))
        
        pair_plot.savefig(file_name)
        plt.clf()

    #helper for scatterplot matrix. Source: https://stackoverflow.com/questions/50832204/show-correlation-values-in-pairplot-using-seaborn-in-python

    def viz_correlation_heat_map(self, experiment_n):
        
        X_selected_features = self.X[:,self.pos==1]
        df = pd.DataFrame(X_selected_features)

        df.columns = self.__get_feature_col_names()

        sns.set_theme(style="white")


        # Compute the correlation matrix
        corr = df.corr()

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        #Save heat map
        heat_map = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        heat_map_fig = heat_map.figure

                   
        
        file_name = "Experiment_Results/Viz/Correlation/heatmap_experiment_{}_{}".format(experiment_n, str(self.config_id))

        
        heat_map_fig.savefig(file_name)
        plt.clf()
        #plt.show() #s_check

    
    #initialises Repeated_Experiment_Results
    def collect_results(self, aggregate_results: Repeated_Experiment_Results):
        return aggregate_results(self.columns, self.optimizer, self.cost,  self.pos, self.selected_features_ratio, FSS_PSO_Experimental_Data_Collector.d, FSS_PSO_Experimental_Data_Collector.cost_histories, self.config_id)


#put back into class, breaks for some reaosn, rename x and y as class gets passed in to x paqrameter for some reason?
def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'ρ = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)
    #write-up: Can compare heat map and scatter plot