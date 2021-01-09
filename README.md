# HelloML
HelloML makes it easy, fast, and intuitive to do Machine Learning (ML).

## Dependencies
```
python>=3.8
```

We recommend creating a `conda` environment before installation to avoid package conflicts.

Install HelloML on your machine by running:
```
pip install -i https://test.pypi.org/simple/ helloml-sideqst
```
Alternatively, you can clone this repository, navigate to the repository's main directory, and run the following command:
```
python setup.py install --user
```
You can now use HelloML from any Python script on your machine by importing it:
```python
from helloml import HelloDataset, HelloModel
```

## Documentation
Full documentation is available at [this repository's wiki](https://github.com/sideqst/helloml/wiki).

For any observations on or issues with HelloML, please contact christine@dsaid.gov.sg.

## How to Use Demo: Predicting survivors on the Titanic shipwreck

Run `titanic_demo.ipynb`.
