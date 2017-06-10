from sklearn import tree
features=[[319,0],[324,0],[129,0],[316,1],[381,0]]
labels=[1,1,1,0,1]
clf=tree.DecisionTreeClassifier()
clf=clf.fit(features,labels)
print (clf.predict([356,1]))
