from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

class DecisionTree:
	def init(self,
     random_seed:int = 42,
     criterion:str = "gini",
     splitter:str = "best",
     max_depth:int = None,
     min_samples_split:int = 2,
     min_samples_leaf:int = 1,
     min_weight_fraction_leaf:float = 0.0,
     max_features:int = None,
     max_leaf_nodes:int = None,
     min_impurity_decrease:float = 0.0,
     class_weight:dict|list|str = None,
     ccp_alpha:float = 0.0) -> None:
		if ccp_alpha < 0.0:
			ccp_alpha = 0.0
   
		self.decisionTreeClassifier = DecisionTreeClassifier(random_state = random_seed)

	def fit(self, X:np.array, y, sample_weight:bool = None, check_input:bool = True):
		pass

	def apply(self, X, check_input:bool = True):
		pass

	def cost_complexity_pruning_path(self, X, y, sample_weight:list|int = None):
		pass

	def decision_path(self, X, check_input:bool = True):
		pass

	def get_depth(self):
		pass

	def get_number_of_leaves(self):
		pass

	def get_parameters(self, deep_copy:bool = True):
		pass

	def get_metadata_routing(self):
		pass

	def predict(self, X, check_input:bool = True):
		pass

	def predict_log_probability(self, X):
		pass

	def predict_proba(self, X, check_input:bool = True):
		pass

	def score(self, X, y, sample_weight:np.array = None):
		pass

	def set_fit_request(self, *, check_input: bool | None | str = "$UNCHANGED$", sample_weight: bool | None | str = "$UNCHANGED$") -> DecisionTreeClassifier:
		pass

	def get_feature_importances(self):
		return self.decisionTreeClassifier.feature_importances_
