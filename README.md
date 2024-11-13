<!-- Final objective: have a table with a MCDropout hyperparameter (like dropout rate, the model precision τ or the number MCDropout iterations), parameters related to the model (like the number of layers), maybe the accuracy of the model, dataset information (like data type, numner of instances, etc.) and finally, as y variable, the (probably average) uncertainty of the model.
Ideally I want to save as much raw data as possible, in order to avoid overhead.

All models should be trained with cross validation, in order to obtain more robust results. We'll start with 3-fold cross validation.

Andiamo a prendere dataset con diversi ratio tra features e size — in maniera tale da prendere un po' di varietà.

Una volta trovata la formula, possiamo andare a fare dei test con delle reti più grosse e vedere se la formula si mantiene.

Potenzialmente potremmo considerare altri tipi di classificatori (tipo random forest).

Analisi che facciamo. Facciamo regressione considerando ogni sample dei dataset come punto di regressione. Quindi, come X abbiamo sia le caratteristiche del dataset, sia gli iperparametri che anche delle caratteristiche del sample. In particolare, possiamo prendere la probabilità che sia outlier, usando IsolationForest; e poi anche una sorta di misura di "omofilia", ovvero quanto un punto è diverso da quello in un suo interno — per fare questo posso addestrare un KNN in maniera supervisionata e andare a fare poi la predizione.


Fare nested cross validation 

Fare correlazione tra incertezza e le variabili di input alla regressione. Eventualmente altri score - anche Mutal Information. 

Ottenere score dalla regressione simbolica.

Nel metodo proporre, vista la formula, mettere un metodo per ridurre l'incertezza. Fare una sorta di inversione. Metteremmo il metodo dopo gli esperimenti. 

Idealmente facciamo vedere che possiamo construire modelli neurali con bordi chiusi. 

Farei calcolo di sample che sono parte della distribuzione del training e che non lo sono.

Quando faccio test, vado a distinguere i punti tra 

1. Tutti i punti in modo omogeneo
2. Tutti outlier
3. Vicino a decision boundary
4. Lontano dal decision boundary


Capire poi come costruire il decisore neurale con superfice chiusa — ci sono dei paper che possiamo prendere (sono funzioni di attivazioni o dei layer fatti apposta). -->

# Order to run scripts

To obtain symbolic regression with multiple cross validation runs, use:
1. `dataset_subsample_selection.py`, to select those datasets that satisfy the conditions defined (large enough).
2. `experiments/prepare_outer_folds/run.py`, to preare the cross validation fold division.
3. `experiments/cc18_subsample_nested/run`, to run the training with varioys hyperparameters; results are saved in the `results_path` variable.
4. `experiments/calculate_uncertainties/run.py`, to calculate the uncertainties over the validation and test folds.
5. `experiments/dataset_features/run.py` to compute dataset features, to add to the hyperparameters for the symbolic regression.
6. `experiments/symbolic_regression/run.py`, to run the symbolic regression over the uncertainties (y) and the hyperparameters + dataset features (X).


