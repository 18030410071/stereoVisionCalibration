
import numpy as np

def kelvinToCelcius(val):
  return (val - 27315)
# the array is loaded into b
kelvin = np.load('outPuts/kelvin/2019_08_05_18_48_23_kelvin.npy')
print(kelvinToCelcius(kelvin))
print(kelvin.shape)
