import os
import pandas as pd
import shutil
import re
import splitfolders
def reorg_dog_data(data_dir, valid_ratio,test_ratio):
    if os.path.exists("classifier/data"):
        shutil.rmtree("classifier/data")
    splitfolders.ratio(data_dir, output="classifier/data",
    seed=1337, ratio=(1.0-valid_ratio-test_ratio, valid_ratio, test_ratio), group_prefix=None, move=False)