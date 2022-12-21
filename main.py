import cv2
img=cv2.imread('pets-2.jpg')

classNames=[]  #importing the coco file
classFile = 'coco.names'
with open(classFile,'rt') as f: #open our file and we need to read it
    classNames=f.read().rstrip('\n').split('\n') #strip it and split it based on newline 
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'  # import the configuration file
weightsPath = 'frozen_inference_graph.pb'   # import the weight file

net = cv2.dnn_DetectionModel(weightsPath,configPath) # create our model, and
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5 ,127.5))
net.setInputSwapRB(True)

classIds, confs, bbox = net.detect(img,confThreshold=0.5)
print(classIds,bbox) # it will give the classids and bbox values

for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox): # 3 informations - classids,confidence and bbox
    cv2.rectangle(img,box,color=(0,255,0),thickness=3)
    cv2.putText(img,classNames[classId-1],(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()