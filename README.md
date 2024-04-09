Final objective: have a table with a MCDropout hyperparameter (like dropout rate, the model precision τ or the number MCDropout iterations), parameters related to the model (like the number of layers), maybe the accuracy of the model, dataset information (like data type, numner of instances, etc.) and finally, as y variable, the (probably average) uncertainty of the model.
Ideally I want to save as much raw data as possible, in order to avoid overhead.

All models should be trained with cross validation, in order to obtain more robust results. We'll start with 3-fold cross validation.

Andiamo a prendere dataset con diversi ratio tra features e size — in maniera tale da prendere un po' di varietà.

