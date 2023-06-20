
import pydicom as dicom
import matplotlib.pylab as plt

path = "C:/Users/Usuario/Downloads/MARIA LUISA ARGAEZ SALCIDO 17 06 23 COMPL/0001.dcm"


from pydicom import dcmread
import pylibjpeg

ds = dcmread(path)
arr = ds.pixel_array

ds = dicom.dcmread(path)

plt.imshow(ds.pixel_array)