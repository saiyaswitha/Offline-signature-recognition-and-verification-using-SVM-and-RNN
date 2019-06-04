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
import numpy as np

def getFeatures(path, img=None, display=False):
    if img is None:
        img = mpimg.imread(path)
    img = preproc.preproc(path, display=display)
    '''img = cv2.imread('bw_image.png')
	img = np.array(img, dtype=np.uint8)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	a=[]
	x_coor=[]
	indices = np.where(gray == [0])
	print (indices)
	coordinates = zip(indices[0], indices[1])
	print (coordinates)
	a=list(zip(indices[0], indices[1]))
	print(len(a))
	print(a[0])
	for j in range(len(a)):  # this loop makes a list of all x coordinates
        x_coor.append((a[j][0]))
	print(x_coor)
	with open("file.txt", "w") as output:
		output.write(str(a))'''
    ratio = features.Ratio(img)
    centroid = features.Centroid(img)
    eccentricity, solidity = features.EccentricitySolidity(img)
    skewness, kurtosis = features.SkewKurtosis(img)
    retVal = (ratio, centroid, eccentricity, solidity, skewness, kurtosis)
    return retVal

def getCSVFeatures(path, img=None, display=False):
    if img is None:
        img = mpimg.imread(path)
    temp = getFeatures(path, display=display)
	
    features = (temp[0], temp[1][0], temp[1][1], temp[2], temp[3], temp[4][0], temp[4][1], temp[5][0], temp[5][1])
    return features

def makeCSV():
    if not(os.path.exists('data/Features')):
        os.mkdir('data/Features')
        print('New folder "Features" created')
    if not(os.path.exists('data/Features/Training')):
        os.mkdir('data/Features/Training')
        print('New folder "Features/Training" created')
    if not(os.path.exists('data/Features/Testing')):
        os.mkdir('data/Features/Testing')
        print('New folder "Features/Testing" created')
    # genuine signatures path
    gpath = 'data/valid/'
    # forged signatures path
    fpath = 'data/invalid/'
    for person in range(1, 56):
        per = (str(person))[-23:]
        print('Saving features for person id-',per)
        
        with open('data/Features/Training/training_'+per+'.csv', 'w') as handle:
            handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')
            # Training set
            for i in range(1,21):
                source = os.path.join(gpath, 'original_'+per+'_'+str(i)+'.png')
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features))+',1\n')
            for i in range(1,21):
                source = os.path.join(fpath, 'forgeries_'+per+'_'+str(i)+'.png')
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features))+',0\n')
        
        with open('data/Features/Testing/testing_'+per+'.csv', 'w') as handle:
            handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')
            # Testing set
            for i in range(21, 25):
                source = os.path.join(gpath, 'original_'+per+'_'+str(i)+'.png')
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features))+',1\n')
            for i in range(21,25):
                source = os.path.join(fpath, 'forgeries_'+per+'_'+str(i)+'.png')
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features))+',0\n')

if __name__ == "__main__":
    #main()
	makeCSV()
	