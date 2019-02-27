import os

"""from DistMat import DistMat
from DTW import DTW
from Metrics import Metrics
from WagnerFischer import WagnerFischer
from KNN import KNN
from Evaluation import Evaluation
from SAX import SAX
from TimeSeries import TimeSeries"""



__author__ = 'Vivek Mahato'
__version__ = "0.0.0.a4"
__bibtex__ = """@misc{mlots,
 title={mlots: A machine learning toolkit for time-series analysis},
 author={Mahato, Vivek},
 year={2019},
 note={\\url{https://github.com/vivekmahato/mlots}}
}"""

on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    import pyximport

    pyximport.install()
