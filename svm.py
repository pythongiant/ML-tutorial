from sklearn import svm
from sklearn import datasets
import matplotlib.pyplot as plt

digits=datasets.load_digits()
#print(len(digits.data))
#print(len(digits.target))
clf=svm.SVC(gamma=0.001,C=100)#C== margin of error
X,y=digits.data[:-10],digits.target[:-10]
clf.fit(X,y)
print(clf.predict(digits.data[9]))
plt.imshow(digits.images[9])
plt.show()