# CE4032

## Installing Dependencies

### Manual Installation
1. [Python 3.6.6 - Python 3.7.5](https://www.python.org/downloads/) - [PSF Licence](https://docs.python.org/3/license.html)

### Required Python Libraries
1. [Pandas](https://pandas.pydata.org/) - [Pandas Licence](https://pandas.pydata.org/pandas-docs/stable/getting_started/overview.html#license)
2. [NumPy](https://numpy.org/) - [NumPy License](https://numpy.org/license.html)
3. [Pytz](https://pypi.org/project/pytz/) - [Pytz License](https://github.com/newvem/pytz/blob/master/LICENSE.txt)
4. [Seaborn](https://seaborn.pydata.org/) - [Seaborn License](https://github.com/mwaskom/seaborn/blob/master/LICENSE)
5. [Scikit-learn](https://scikit-learn.org/stable/) - [Scikit License](https://github.com/scikit-learn/scikit-learn/blob/master/COPYING)
6. [SciPy](https://www.scipy.org/index.html) - [SciPy License](https://www.scipy.org/scipylib/license.html)
7. [Tensorflow](https://www.tensorflow.org/) - [Tensorflow License](https://github.com/tensorflow/tensorflow/blob/master/LICENSE)
8. [Tqdm](https://github.com/tqdm/tqdm) - [Tqdm License](https://github.com/tqdm/tqdm/blob/master/LICENCE)
9. [Folium](https://python-visualization.github.io/folium/) - [Folium License](https://github.com/python-visualization/folium/blob/master/LICENSE.txt)
10. [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) - [JupyterLab License](https://github.com/jupyterlab/jupyterlab/blob/master/LICENSE)
11. [XGBoost](https://xgboost.readthedocs.io/en/latest/) - [XGBoost License](https://github.com/dmlc/xgboost/blob/master/LICENSE)

### Installation Steps
1. Once Python has been installed, input in _cmd_ : `python -m pip install -r requirements.txt`

### Launch Project
1. After installing the dependencies from the requirements, go to [ECML/PKDD 15: Taxi Trip Time Prediction (II)](https://www.kaggle.com/c/pkdd-15-taxi-trip-time-prediction-ii/data) and download the dataset.
2. Place the downloaded dataset into the directory datasets/
3. Run 'data_creation.py' with the command `python data_creation.py`. This will take about 4 hours to generate the dataset after feature engineering that will be used for training purposes and for visualization.
4. Any python files are expected to be run standalone. This means that to run the python files you have to navigate to the relevant directory and run it. For example to run linear_regression.py you navigate to the training directory and run the following command `python linear_regression.py`
5. To run the Jupyter notebooks you have to start the Jupyter notebook server first. Run the following command in the command line: `jupyter notebook` to start the Jupyter notebook server. Afterwards, proceed to navigate to the training directory from the browser and run the Jupyter notebooks.
