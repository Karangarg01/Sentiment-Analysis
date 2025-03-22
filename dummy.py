from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("Moodclassifier.h5")

print(model.input_shape)

