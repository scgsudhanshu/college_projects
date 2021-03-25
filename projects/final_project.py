# Importing all the libraries
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import os

#haarcascade classifier
face_classifier = cv2.CascadeClassifier('C:\ProgramData\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

# recognizer 
recognizer=cv2.face_LBPHFaceRecognizer.create(1,8,8,8)

#function for face_detection
def face_detection(image):
   # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = face_classifier.detectMultiScale(image, scaleFactor=1.2,minNeighbors=3)
    for (x,y,w,h) in face:
        image=image[y:y+w,x:x+h]
    return image

#collecting samples
def collect_samples():
    source=input("Enter the source:")
    try:
        source=int(source)
    except:
        source=str(source)
    cap=cv2.VideoCapture(source)
    label=input("Enter the Name:")
    path='D:/face_rec/{}'.format(label)
    try:
        os.mkdir(path)
    except:
        print("ALready Exist")
    while(True):
        ret,frame=cap.read()
       # frame=cv2.resize(frame,(400,640))
        
        #frame=cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
        frame_img=face_detection(frame)
        i=0
        for i in range(100000):
            if i%1000==0:
                cv2.imwrite('D:/face_rec/{}/{}_{}.jpg'.format(label,label,i),frame_img)
                if(i==99000):
                    cv2.putText(frame,"Samples Collected",(10,20),1,2,(0,255,0),2)
        cv2.imshow("sample",frame)
        if cv2.waitKey(200) & 0xFF==ord("q"):
            break
    cv2.destroyAllWindows()
    cap.release()
    return path


# training function
def training():
    dir=collect_samples()
    dir="{}/*.*".format(dir)
    training_files=glob(dir)
    print(training_files)
    training_data=[]
    ids=[]
    for f1 in tqdm(training_files):
        img=cv2.imread(f1,cv2.IMREAD_GRAYSCALE)
        #img=cv2.resize(img,(150,150))
        img=face_detection(img)
        
        #img=cv2.resize(img,(150,150))
        
        
        print(f"processing file:{f1}")
        training_data.append(np.array(img))
        label=f1.split('\\')[1].split('_')[0]      
        ids.append(0)
    data_trained=np.array(training_data)
    print(f"shape:{data_trained.shape}")
    print(data_trained)
    ids=np.array(ids)
    ids=ids.astype('int32')
    print(ids)
    print(data_trained.shape)
    recognizer=cv2.face.LBPHFaceRecognizer_create(1,8,8,8)
    #try:
    
    recognizer.train(training_data,ids)
    #except:
    #   print("size not find")
    file_name='D:/face_rec/models/{}.yml'.format(label)
    
    recognizer.write(file_name)
    print("Model has been saved to : {}".format(file_name))

def face_matching(frame_gray):    
    
    flag=0
    test_img=face_detection(frame_gray)
    dir='D:/face_rec/models/*.*'
    models=glob(dir)
    for f1 in models:
        recognizer.read(f1)
        label=f1.split('\\')[1].split('.')[0]
        recognizer.read('D:/face_rec/models/{}.yml'.format(label))
        print(f"Recognizer is {label}")
        pred_label,confidence=recognizer.predict(test_img)
        print(f"{pred_label},{confidence}")
        if confidence<=60 :
            flag=1              
            break
        else:
            continue
    
    return flag,label

def testing():
    flag=-1
    label=''
    source=input("Enter the source:")
    try:
        source=int(source)
    except:
        source=str(source)
    vid=cv2.VideoCapture(source)
    while(True):
        ret,frame=vid.read()
        
      



        frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        loc=face_classifier.detectMultiScale(frame,1.2,3)
        for (x,y,w,h) in loc:
            start=(x,y)
            end=(x+w,y+h)
            color=(255,255,255)
            thickness=2
            cv2.rectangle(frame,start,end,color,thickness)
        
            

        if flag==-1:
            cv2.putText(frame,"Processing.....".format(label),(100,150),1,2,(0,255,0),2)

            flag,label=face_matching(frame_gray)
        if flag==1:
            cv2.putText(frame,"Match Found {}".format(label),(100,150),1,2,(0,255,0),2)
            print(f"Match Found: {label}")
            

        elif flag==0:
            cv2.putText(frame,"No Match Found".format(label),(100,150),1,2,(0,255,0),2)
            print(" No Match Found")
        
            

        if cv2.waitKey(50)& 0xFF==ord('q'):
            break
        cv2.imshow("frame",frame)

    vid.release()
    cv2.destroyAllWindows()
    return flag

#collect_samples()
#training()
flag=testing()
if flag==0:
    print("NEW PERSON ")
    training()

