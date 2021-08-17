def OF_equation_scorer(obj_1, obj_2, alpha):
    #print(obj_1)
    j = (alpha * (1-obj_1) + (1.0 - alpha) * (obj_2))
    return j

def OF_equation_mse(obj_1, obj_2, alpha):
    #print(obj_1)
    j = (alpha * (obj_1) + (1.0 - alpha) * (obj_2))
    return j

def setup_pso_builder(config, data, builder):

    regressor = None

    optimisation_models = {
        'LR': linear_model.LinearRegression(),
        'RFR': RandomForestRegressor(n_estimators=2,max_depth=2) 
    }

    if config['optimisation_model'] in optimisation_models:
        regressor = optimisation_models[config['optimisation_model']]

    m1 = None
    m2 = None

    evaluation_models = {
        'LR': linear_model.LinearRegression(),
        'RFR': RandomForestRegressor(n_estimators=2,max_depth=2) 
    }

    if config['eval_model_post_optimisation'] in evaluation_models:
        m1 = evaluation_models[config['eval_model_post_optimisation']]
        m2 = evaluation_models[config['eval_model_post_optimisation']]

    #set OF equation based upon this
    performance_metrics = {
        'R2': R2,
        'MSE': mse

    }

    objective_fcns = {
        'R2': OF_equation_scorer,
        'MSE': OF_equation_mse
    }

    if config['performance_metric'] in performance_metrics:
        performance_metric = evaluation_models[config['performance_metric']]
        obj_fcn = objective_fcns[config['performance_metric']]

    n_particles = config['n_particles']
    iterations = config['iterations']

    keys_to_extract = ["c1", "c2", "w", "k", "p"]
    
    options = {key: config[key] for key in keys_to_extract}

    a = builder(data, options, n_particles, iterations, regressor,performance_metric, config['alpha_balancing_coefficient'], obj_fcn, m1, m2 )


    return a   

    









def run(config):
    
    for x in range(config['runs']):

        #Mandatory function calls

        #partially initialise
        a = setup_pso_builder(parameter_config, data, FSS_PSO_Builder)
        #a = FSS_PSO_Builder(options, data, 100,2,regressor,mse, 0.5,OF_equation_mse, m1,m2)
        #initialise other fields
        a.count_selected_features()
        a.do_cross_val()
        #add results
        a.create_results()

        #optional function calls

        if config['save_scatter_plot_matric']:
            a.viz_cost_history()

        if config['save_scatter_plot_matrix']:
            a.viz_scatter_plot_matrix()

        #reset optimiser for next run of PSO
        a.optimizer.reset()
    return a


parameter_config = {
    'runs': 5, #each run will produce a subset
    #OF Parameters
    'optimisation_model': 'LR',
    'performance_metric': 'R2',
    'eval_model_post_optimisation':'LR',
    'alpha_balancing_coefficient': 0.5,
    #pso optimiser swarm parameters
    'n_particles':30,
    'iterations':5,
    #pso hyperparameters
    'c1': 0.5, 'c2': 0.5, 'w':0.3, 'k': 30, 'p':2,
    #optional functionality parameters
    'save_performance_cost': False,
    'save_scatter_plot_matrix': False,
    'save_evaluation_results': True,
    'plot_subset_size_histo': True,
    'plot_feature_frequency': True,
    
}
def excel_config_parser: pass #should return a pandas dataframe, which gets passed to 'setup_pso_builder' to be used in 'builder'
data = excel_config_parser()


def aggregate_optional_functions(config, post_runs):
    aggregate_of_runs = post_runs.build()

    aggregate_of_runs.save_results_csv()

    if config['plot_subset_size_histo']:
        aggregate_of_runs.plot_subset_size_hist()
    if config['plot_feature_frequency']:
        aggregate_of_runs.


a = run(parameter_config)

aggregate_optional_functions(parameter_config, a)

#Todo
#read config file that runs PSO algorithm
    #do logic
    #add excel sheet functionality
        #each row in excel sheet is a 'job'
#Modularise code
    #read up on python modules

#Functionality: save into folders categorised by module

#Read
    #General articles: evaluation metrics, ensemble methods
    #Journals
#Other
    #apply for extension