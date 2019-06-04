import preproc
import features
import os
import cv2
from featuresgrid import gridfeatures
from dataset import getCSVFeatures

def testing(path):
    glofe = dataset.getCSVFeatures(path)
    features = getlocalFeatures(path)
    if not(os.path.exists('data/TestFeatures')):
        os.mkdir('data/TestFeatures')
    with open('data/TestFeatures/testcsv.csv', 'w') as handle:
        handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,11,12,13,14,15,16,21,22,23,24,25,26,31,32,33,34,35,36,41,42,43,44,45,46,51,52,53,54,55,56,61,62,63,64,65,66,71,72,73,74,75,76,81,82,83,84,85,86,91,92,93,94,95,96\n')
        handle.write(','.join(map(str, glofe))+',')
        handle.write(','.join(map(str, features))+'\n')				
				


if __name__=="__main__":
    main()