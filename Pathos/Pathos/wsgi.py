"""
WSGI config for pathos project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.0/howto/deployment/wsgi/
"""

import os
from django.core.wsgi import get_wsgi_application
# import json as j
# import pandas as pd
# import re
# import numpy as np
# from nltk.corpus import stopwords
# from nltk.stem import SnowballStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
# from sklearn.feature_selection import SelectKBest, chi2

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "pathos.settings")

application = get_wsgi_application()


