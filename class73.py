from keras.models import load_model
import cv2
import numpy as np

myModel=load_model("keras_model.h5",compile=False)

classNames=open("labels.txt","r").readlines()
# print(classNames)

camera=cv2.VideoCapture(0)

while True:
    dummy,img=camera.read()

    img_flipped=cv2.flip(img,1)
    img_flipped=cv2.resize(img_flipped,(224,224),interpolation=cv2.INTER_AREA)

    cv2.imshow("prediction",img_flipped)    
    img_flipped=np.asarray(img_flipped,dtype=np.float32).reshape(1,224,224,3)
    # 0-255
    img_flipped=(img_flipped/127.5) -1

    prediction=myModel.predict(img_flipped) 
    # print(prediction)

    index=np.argmax(prediction)
    className=classNames[index]
    confScore=prediction[0][index]

    print(className,confScore*100)
    if(cv2.waitKey(25) == 27):
        break

camera.release()
cv2.destroyAllWindows()