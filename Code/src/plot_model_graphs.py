import os
from tensorflow.keras.utils import plot_model
from ml_base import models


save_path = "../../Data/Output_Data/graphs/"


""" # define single view model
base_model_single_xpl = models.define_base_model_single_xpl(learning_rate=0.001)
plot_model(base_model_single_xpl, os.path.join(save_path, "base_model_single_xpl.png"))

# define 2 view models
base_model_early_fusion_2_view = models.define_base_model_early_fusion_2_view(learning_rate=0.001)
plot_model(base_model_early_fusion_2_view, os.path.join(save_path, "base_model_early_fusion_2_view.png"))

base_model_mid_fusion_2_view = models.define_base_model_mid_fusion_2_view(learning_rate=0.001)
plot_model(base_model_mid_fusion_2_view, os.path.join(save_path, "base_model_mid_fusion_2_view.png")) """


""" # define 3 view models
base_model_early_fusion_3_view = models.define_base_model_early_fusion_3_view(learning_rate=0.001)
plot_model(base_model_early_fusion_3_view, os.path.join(save_path, "base_model_early_fusion_3_view.png")) """

""" base_model_mid_fusion_3_view = models.define_base_model_mid_fusion_3_view(learning_rate=0.001)
plot_model(base_model_mid_fusion_3_view, os.path.join(save_path, "base_model_mid_fusion_3_view.png")) """

""" # define 6 view models
base_model_early_fusion_6_view = models.define_base_model_early_fusion_6_view(learning_rate=0.001)
plot_model(base_model_early_fusion_6_view, os.path.join(save_path, "base_model_early_fusion_6_view.png")) """

base_model_mid_fusion_6_view = models.define_base_model_mid_fusion_6_view(learning_rate=0.001)
plot_model(base_model_mid_fusion_6_view, os.path.join(save_path, "base_model_mid_fusion_6_view.png"))
