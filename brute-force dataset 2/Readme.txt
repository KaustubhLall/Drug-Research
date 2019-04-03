The file is run by running python main.py, ensuring you use python3.

The dependencies are:
sklearn

File descriptions:

	1. main.py :
	Code to run the files. Ensure test, training data are in the same format as given in the github.
	Call enumerate_over_features('test.csv', 'train.csv', [1, 2, 3, ...]) where the list contains the number of features to try.
	Other functions are called by this main function.

	2. gen_combinations.py
	Contains code to generate sequences of combinations chosen from a collection of items.

	3. classifier.py
	Contains code to find AUCs for all classifiers. Each function runs a single classifier.

	4. datacontainer.py
	Contains code to define a custom datacontainer object. This is used with gen_combinations to quickly subset from the main dataset over a sequence of given columns.

	5. parse_results.py
	Once main.py runs and generates results files, parse results will take the top results and compile them into a csv file for easy visualization and summary.
	Call find_best_over(number of top results to select, [list of features to try]). For example if you ran main.py with [2, 5, 6, 7], your list of features to try would be [2, 5, 6, 7] as well.

All code is commented with specific descriptions right under the method declarations. Not all code is completed and not every function is used, it is designed in a way such that new features should be easy to implement in the future.

	
	