import pickle
import numpy as np

try:
    model = pickle.load(open('model.pkl', 'rb'))
    features = np.array([[18, 75, 2, 2]])
    pred = model.predict(features)
    label = "Dropout" if pred[0] == 1 else "Continue"
    print("prediction:", pred, "->", label)
except Exception as e:
    print("ERROR:", e)
