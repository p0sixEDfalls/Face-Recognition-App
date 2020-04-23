from keras.models import load_model
from keras.utils.vis_utils import plot_model
import shared

model = load_model(shared.MODEL_NAME)
print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)