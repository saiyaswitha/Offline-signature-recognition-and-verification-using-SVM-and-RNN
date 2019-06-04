import preproc
import features
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from scipy import ndimage
from skimage.measure import regionprops
from skimage import io
from skimage.filters import threshold_otsu   # For finding the threshold for grayscale to binary conversion
import cv2
from featuresgrid import gridfeatures
import math
from dataset import getCSVFeatures

def getlocalFeatures(path, img=None, display=False):
	if img is None:
		img = mpimg.imread(path)
	img = preproc.preproc(path, display=display)
	img = np.array(img, dtype=np.uint8)
	#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	a=[]
	x_coor=[]
	indices = np.where(img !=[0])
	#print (indices)
	coordinates = zip(indices[0], indices[1])
	#print (coordinates)
	a=list(zip(indices[0], indices[1]))
	gridfe=gridfeatures(a)
	'''print(len(a))
	print(a[0])
	for j in range(len(a)):  # this loop makes a list of all x coordinates
        x_coor.append((a[j][0]))
	print(x_coor)
	with open("file.txt", "w") as output:
		output.write(str(a))
    ratio = features.Ratio(img)
    centroid = features.Centroid(img)
    eccentricity, solidity = features.EccentricitySolidity(img)
    skewness, kurtosis = features.SkewKurtosis(img)
    retVal = (ratio, centroid, eccentricity, solidity, skewness, kurtosis)'''
	return gridfe

'''def getCSVFeatures(path, img=None, display=False):
    if img is None:
        img = mpimg.imread(path)
    temp = getFeatures(path, display=display)
    features = (temp[0], temp[1][0], temp[1][1], temp[2], temp[3], temp[4][0], temp[4][1], temp[5][0], temp[5][1])
    return features'''

def makeCSV():
    if not(os.path.exists('data/Prep')):
        os.mkdir('data/Prep')
        print('New folder "Prep" created')
    if not(os.path.exists('data/Prep/Valid')):
        os.mkdir('data/Prep/Valid')
        print('New folder "Prep/Valid" created')
    if not(os.path.exists('data/Prep/Invalid')):
        os.mkdir('data/Prep/Invalid')
        print('New folder "Prep/Invalid" created')
    # genuine signatures path
    gpath = 'data/valid/'
    # forged signatures path
    fpath = 'data/invalid/'
    '''for person in range(1, 56):
        per = (str(person))[-23:]
        print('Saving features for person id-',per)
        
        with open('data/Prep/Valid/pre_'+per+'.txt', 'w') as handle:
            #handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')
            # Training set
            for i in range(1,25):
                source = os.path.join(gpath, 'original_'+per+'_'+str(i)+'.png')
                features = getFeatures(path=source)
                handle.write(str(features)+'\n')
            
        
        with open('data/Prep/Invalid/pre_'+per+'.txt', 'w') as handle:
            for i in range(1,25):
                source = os.path.join(fpath, 'forgeries_'+per+'_'+str(i)+'.png')
                features = getFeatures(path=source)
                handle.write(str(features)+'\n')'''
    for person in range(1, 56):
        per = (str(person))[-23:]
        print('Saving features for person id-',per)
        
        '''with open('data/Features/Training/training_'+per+'.csv', 'w') as handle:
            handle.write('class_label,ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,11,12,13,14,15,16,21,22,23,24,25,26,31,32,33,34,35,36,41,42,43,44,45,46,51,52,53,54,55,56,61,62,63,64,65,66,71,72,73,74,75,76,81,82,83,84,85,86,91,92,93,94,95,96\n')
            # Training set
            for i in range(1,25):
                source = os.path.join(gpath, 'original_'+per+'_'+str(i)+'.png')
                glofe=getCSVFeatures(path=source)
                handle.write(''.join(map(str, per))+',')
                handle.write(','.join(map(str, glofe))+',')
                features = getlocalFeatures(path=source)
                handle.write(','.join(map(str, features))+'\n')
            for i in range(1,21):
                source = os.path.join(fpath, 'forgeries_'+per+'_'+str(i)+'.png')
                glofe=getCSVFeatures(path=source)
                handle.write(','.join(map(str, per)))
                handle.write(','.join(map(str, glofe))+',')
                features = getlocalFeatures(path=source)
                handle.write(','.join(map(str, features))+'\n')'''
        
        with open('data/forged.csv', 'a') as handle:
            handle.write('class_label,ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,111,121,131,141,151,161,171,181,112,122,132,142,152,162,172,182,113,123,133,143,153,163,173,183,114,124,134,144,154,164,174,184,115,125,135,145,155,165,175,185,1_px,211,221,231,241,251,261,271,281,212,222,232,242,252,262,272,282,213,223,233,243,253,263,273,283,214,224,234,244,254,264,274,284,215,225,235,245,255,265,275,285,2_px,311,321,331,341,351,361,371,381,312,322,332,342,352,362,372,382,313,323,333,343,353,363,373,383,314,324,334,344,354,364,374,384,315,325,335,345,355,365,375,385,3_px,411,421,431,441,451,461,471,481,412,422,432,442,452,462,472,482,413,423,433,443,453,463,473,483,414,424,434,444,454,464,474,484,415,425,435,445,455,465,475,485,4_px,511,521,531,541,551,561,571,581,512,522,532,542,552,562,572,582,513,523,533,543,553,563,573,583,514,524,534,544,554,564,574,584,515,525,535,545,555,565,575,585,5_px,611,621,631,641,651,661,671,681,612,622,632,642,652,662,672,682,613,623,633,643,653,663,673,683,614,624,634,644,654,664,674,684,615,625,635,645,655,665,675,685,6_px,711,721,731,741,751,761,771,781,712,722,732,742,752,762,772,782,713,723,733,743,753,763,773,783,714,724,734,744,754,764,774,784,715,725,735,745,755,765,775,785,7_px,811,821,831,841,851,861,871,881,812,822,832,842,852,862,872,882,813,823,833,843,853,863,873,883,814,824,834,844,854,864,874,884,815,825,835,845,855,865,875,885,8_px,911,921,931,941,951,961,971,981,912,922,932,942,952,962,972,982,913,923,933,943,953,963,973,983,914,924,934,944,954,964,974,984,915,925,935,945,955,965,975,985,9_px\n')
            # Testing set
            for i in range(1, 25):
                source = os.path.join(fpath, 'forgeries_'+per+'_'+str(i)+'.png')
                glofe=getCSVFeatures(path=source)
                handle.write(''.join(map(str, per))+',')
                handle.write(','.join(map(str, glofe))+',')
                features = getlocalFeatures(path=source)
                handle.write(','.join(map(str, features))+'\n')
            '''for i in range(21,25):
                source = os.path.join(fpath, 'forgeries_'+per+'_'+str(i)+'.png')
                glofe=getCSVFeatures(path=source)
                handle.write(','.join(map(str, per))+',')
                handle.write(','.join(map(str, glofe))+',')
                features = getlocalFeatures(path=source)
                handle.write(','.join(map(str, features))+'\n')'''

if __name__ == "__main__":
    #main()
	makeCSV()
	
			