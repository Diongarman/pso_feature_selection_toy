import glob
# Import modules
import pandas as pd

# sklearn imports
# learning algos
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
# metrics
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
#internal imports
from feat_select.feature_selection_regression import Repeated_Experiment_Results, FSS_PSO_Experimental_Data_Collector #Is interaction/dependency between these classes




#GLOBALS
candidate_solutions = {}
# data = pd.read_excel('drivPoints.xlsx')
#def excel_config_parser: pass #should return a pandas dataframe, which gets passed to 'setup_pso_builder' to be used in 'builder'
#data = excel_config_parser()
from sklearn.datasets import make_regression,make_sparse_uncorrelated
X, y = make_regression(n_samples=2000, n_features=25, bias=0,
                           n_informative=9,
                           effective_rank=3,
                           random_state=1)

#X, y = make_sparse_uncorrelated(n_samples=1000, n_features=50)
# Plot toy dataset per feature
#
data = pd.DataFrame(X)
data['labels'] = y

print(data)



class Experiment_Config_Manager:
    def __init__(self, data, config, builder):
        self.data = data
        self.config = config
        self.builder = builder
        import uuid
        self.config_uid = uuid.uuid1() #denotes and instance of a particular experiment configuration

    def __setup_pso_builder(self):

    

        optimisation_models = {
            'LR': linear_model.LinearRegression(),
            'RFR': RandomForestRegressor() 
        }

        if self.config['wrapper_model'] in optimisation_models:
            regressor = optimisation_models[self.config['wrapper_model']]



        evaluation_models = {
            'LR': linear_model.LinearRegression(),
            'RFR': RandomForestRegressor(n_estimators=2,max_depth=2) 
        }

        if self.config['eval_model_post_optimisation'] in evaluation_models:
            m1 = evaluation_models[self.config['eval_model_post_optimisation']]
            m2 = evaluation_models[self.config['eval_model_post_optimisation']]

        #set OF equation based upon this
        performance_metrics = {
            'R2': r2,
            'MSE': mse

        }

        objective_fcns = {
            'R2': self.__OF_equation_scorer,
            'MSE': self.__OF_equation_mse
        }

        if self.config['performance_metric'] in performance_metrics:
            performance_metric = performance_metrics[self.config['performance_metric']]
            obj_fcn = objective_fcns[self.config['performance_metric']]


        #Todo: How to tune these parameters?
        n_particles = self.config['n_particles'] 
        iterations = self.config['iterations']


        keys_to_extract = ["c1", "c2", "w", "k", "p"]
        
        options = {key: self.config[key] for key in keys_to_extract}

        a = self.builder(self.data, options, n_particles, iterations, regressor, performance_metric, self.config['alpha_balancing_coefficient'], obj_fcn, m1, m2, self.config['pso_parameter_optimiser'],self.config_uid )

        return a


    def __OF_equation_scorer(self, obj_1, obj_2, alpha):
        #print(obj_1)
        j = (alpha * (1-obj_1) + (1.0 - alpha) * (obj_2))
        return j        
    
    def __OF_equation_mse(self, obj_1, obj_2, alpha):
        #print(obj_1)
        j = (alpha * (obj_1) + (1.0 - alpha) * (obj_2))
        return j

    def run_experiment_n_times(self):


        for n in range(self.config['runs']):

            #Mandatory function calls

            #partially initialise
            a = self.__setup_pso_builder() #Todo: add decoupling later
            #initialise other fields
            a.count_selected_features()
            a.wrapper_validation_cross_val() #Todo: Does this need to be conditional
            #add results
            a.create_results()
            #save cost_history
            a.save_cost_history()
            #optional function calls

            if self.config['save_performance_cost']:
                a.viz_cost_history(n)

            if self.config['save_scatter_plot_matrix']:
                a.viz_scatter_plot_matrix(n)
            if self.config['save_correlation_matrix']:
                a.viz_correlation_heat_map(n)
            

            #reset optimiser for next run of PSO
            a.optimizer.reset()
        return a

experiment_parameter_config = {
    'runs': 1, #each run will produce a subset
    #OF Parameters
    'wrapper_model': 'LR',
    'performance_metric': 'R2',
    'eval_model_post_optimisation':'LR',
    'alpha_balancing_coefficient': 0.5,
    #pso optimiser swarm parameters
    'n_particles':50,
    'iterations':100,
    #pso hyperparameters
    'pso_parameter_optimiser':'random_search',
    #Todo: Make sure to signify meaning behind coefficients e.g. exploitation vs exploration
    'c1': 0.5, 'c2': 0.5, 'w':0.3, 'k': 30, 'p':2,
    #optional functionality parameters
    'save_performance_cost': True,
    'save_scatter_plot_matrix': True,
    'save_correlation_matrix': True,
    'save_evaluation_results': True,
    'plot_subset_size_histo': True,
    'plot_feature_frequency': True,
    
}


pso_ui = Experiment_Config_Manager(data, experiment_parameter_config, FSS_PSO_Experimental_Data_Collector)
multiple_runs = pso_ui.run_experiment_n_times()# Aggregate data
aggregated_PSO_results = multiple_runs.collect_results(Repeated_Experiment_Results)
aggregated_PSO_results.save_results_csv()
aggregated_PSO_results.aggregate_optional_functions(experiment_parameter_config)



# #Todo
# #read config file that runs PSO algorithm
#     #do logic
#     #add excel sheet functionality
#         #each row in excel sheet is a 'job'


# #Functionality: save into folders categorised by model

# #Read
#     #General articles: evaluation metrics, ensemble methods
#     #Journals
# #Other
#     #apply for extension


# #https://opensource.com/article/18/5/how-retrieve-source-code-python-functions