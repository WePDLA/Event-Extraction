import numpy as np
import joblib



a = np.zeros([5,5])
b = a[:3,:3]
joblib.dump(a, 'a.pkl')
print(b)