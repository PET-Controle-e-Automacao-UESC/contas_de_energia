from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

class DecisionTree:
	def init(self, random_seed:int = 42):
		self.decisionTreeClassifier = DecisionTreeClassifier(random_state = random_seed)

	def fit(X, y, sample_weight:bool = None, check_input:bool = True):
		pass

	def apply(X, check_input:bool = True):
		pass

	def cost_complexity_pruning_path(X, y, sample_weight = None):
		pass

	def decision_path(X, check_input:bool = True):
		pass

	def get_depth():
		pass

	def get_number_of_leaves():
		pass

	def get_parameters(deep_copy:bool = True):
		pass

	def get_metadata_routing():
		pass

	def predict(X, check_input:bool = True):
		pass

	def predict_log_probability(X):
		pass

	def predict_proba(X, check_input:bool = True):
		pass

	def score(X, y, sample_weight:np.array = None):
		pass

	def set_fit_request(*, check_input: bool | None | str = "$UNCHANGED$", sample_weight: bool | None | str = "$UNCHANGED$") -> DecisionTreeClassifier:
		pass

	def get_feature_importances():
		return self.decisionTreeClassifier.feature_importances_
