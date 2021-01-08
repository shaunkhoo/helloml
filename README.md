# helloml
Simple ML for beginners.

## Dependencies
```
python>=3.8
```

Install as a package by running:
```
pip install -i https://test.pypi.org/simple/ helloml-sideqst
```

## Documentation
You can use the following methods to run a basic ML workflow:
```python
from helloml import HelloDataset, HelloML

# Loading data
data = HelloDataset()
data.load('path/to_your/data.csv')

# Exploring data
data.explore()

# Engineering features using arithmetic operations
data.sum_feature('new_feature_name', feature_list=['Feature1', 'Feature2'])
data.subtract_feature('new_feature_name', 'Feature1', 'Feature2')
data.multiply_feature('new_feature_name', feature_list=['Feature1', 'Feature2'])
data.divide_feature('new_feature_name', 'FeatureAtNumerator', 'FeatureAtDenominator')

# Cleaning data
data.drop_feature(['Feature1', 'Feature2', 'Feature3'])
data.convert_to_numerical(['Feature1', 'Feature2', 'Feature3'])

# Set the target feature
data.set_target_feature('Survived')

# Set the model
model = HelloModel('Logistic Regression')
model = HelloModel('Decision Tree')
model = HelloModel('K-Nearest Neighbours')

# Train on loaded dataset
model.train(data)

# Test on loaded dataset
model.test(data)
```

For any observations on or issues with HelloML, please contact christine@dsaid.gov.sg.

## Demo: Predicting survivors on the Titanic shipwreck

Run `titanic_demo.ipynb`.
