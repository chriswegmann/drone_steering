import os
currpath = os.getcwd()
content = os.listdir()

from app_local.module import DataEnsembler, DataResampler, GestureTransformer

de = DataEnsembler(120)
de.investigate_available_datafiles(data_dir = "data/gesture/")
de.load_data()
de.rescale_data_frames(time_of_first_frame='avg', verbose=True)