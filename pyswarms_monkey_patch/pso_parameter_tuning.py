from pyswarms.utils.search.grid_search import GridSearch
from pyswarms.utils.search.random_search import RandomSearch

class RandomSearchUpdate(RandomSearch):
    
    def __init__(        
        self,
        optimizer,
        n_particles,
        dimensions,
        options,
        objective_func,
        iters,
        n_selection_iters,
        bounds=None,
        velocity_clamp=(0, 1),
        **kwargs
    ):
        
        super(RandomSearchUpdate,self).__init__(
            optimizer,
            n_particles,
            dimensions,
            options,
            objective_func,
            iters,
            n_selection_iters,
            bounds=bounds,
            velocity_clamp=velocity_clamp,
        )
        self.kwargs = kwargs
        # invoke assertions
        self.assertions()
    
    def generate_score(self, options):
        """Generate score for optimizer's performance on objective function

        Parameters
        ----------

        options: dict
            a dict with the following keys: {'c1', 'c2', 'w', 'k', 'p'}
        """

        # Intialize optimizer
        f = self.optimizer(
            self.n_particles, self.dims, options, self.bounds, velocity_clamp=self.vclamp
        )
        
        #print(self.kwargs)
        #breakpoint()

        # Return score
        return f.optimize(self.objective_func, iters = self.iters,**self.kwargs)
    
    def search(self, maximum=False):
        import operator as op
        """Compare optimizer's objective function performance scores
        for all combinations of provided parameters

        Parameters
        ----------

        maximum: bool
            a bool defaulting to False, returning the minimum value for the
            objective function. If set to True, will return the maximum value
            for the objective function.
        """

        # Generate the grid of all hyperparameter value combinations
        grid = self.generate_grid()

        # Calculate scores for all hyperparameter combinations
        scores = [self.generate_score(i)[0] for i in grid]
        
        print(min(scores))
        print(len(scores))

        # Default behavior
        idx, self.best_score = min(enumerate(scores), key=op.itemgetter(1))

        # Catches the maximum bool flag
        if maximum:
            idx, self.best_score = max(enumerate(scores), key=op.itemgetter(1))

        # Return optimum hyperparameter value property from grid using index
        self.best_options = op.itemgetter(idx)(grid)
        return self.best_score, self.best_options

class GridSearchUpdate(GridSearch):
    
    def __init__(        
        self,
        optimizer,
        n_particles,
        dimensions,
        options,
        objective_func,
        iters,
        bounds=None,
        velocity_clamp=(0, 1),
        **kwargs
    ):
        
        super(GridSearchUpdate,self).__init__(
            optimizer,
            n_particles,
            dimensions,
            options,
            objective_func,
            iters,
            bounds=bounds,
            velocity_clamp=velocity_clamp,
        )
        self.kwargs = kwargs
        # invoke assertions
        self.assertions()
    
    def generate_score(self, options):
        """Generate score for optimizer's performance on objective function

        Parameters
        ----------

        options: dict
            a dict with the following keys: {'c1', 'c2', 'w', 'k', 'p'}
        """

        # Intialize optimizer
        f = self.optimizer(
            self.n_particles, self.dims, options, self.bounds, velocity_clamp=self.vclamp
        )
        
        #print(self.kwargs)
        #breakpoint()

        # Return score
        return f.optimize(self.objective_func, iters = self.iters,**self.kwargs)
    
    def search(self, maximum=False):
        import operator as op
        """Compare optimizer's objective function performance scores
        for all combinations of provided parameters

        Parameters
        ----------

        maximum: bool
            a bool defaulting to False, returning the minimum value for the
            objective function. If set to True, will return the maximum value
            for the objective function.
        """

        # Generate the grid of all hyperparameter value combinations
        grid = self.generate_grid()

        # Calculate scores for all hyperparameter combinations
        scores = [self.generate_score(i)[0] for i in grid]
        
        print(min(scores))
        print(len(scores))

        # Default behavior
        idx, self.best_score = min(enumerate(scores), key=op.itemgetter(1))

        # Catches the maximum bool flag
        if maximum:
            idx, self.best_score = max(enumerate(scores), key=op.itemgetter(1))

        # Return optimum hyperparameter value property from grid using index
        self.best_options = op.itemgetter(idx)(grid)
        return self.best_score, self.best_options
