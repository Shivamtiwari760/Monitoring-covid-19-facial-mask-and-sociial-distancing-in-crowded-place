from mylib import config, thread
from mylib.mailer import Mailer
from mylib.detection import detect_people
from imutils.video import VideoStream, FPS
from scipy.spatial import distance as dist
import numpy as np
#import gpsloc as g
import requests
import os
import pandas as pd
from datetime import datetime
from playsound import playsound
import argparse, imutils, cv2, os, time, schedule
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()

harcascadePath = "haarcascade_frontalface_default.xml"
faceCascade1 = cv2.CascadeClassifier(harcascadePath);

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
cascPath = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(harcascadePath)
model = load_model("mask_recog_ver2.h5")
# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

#----------------------------Parse req. arguments------------------------------#
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args())
#------------------------------------------------------------------------------#
def process():
        

        # load our YOLO object detector trained on COCO dataset (80 classes)
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        recognizer.read("TrainingImageLabel\Trainner.yml")
        mask=0.0
        withoutMask=0.0
        bb=""
        
        # check if we are going to use GPU
        if config.USE_GPU:
                # set CUDA as the preferable backend and target
                print("")
                print("[INFO] Looking for GPU")
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

        # if a video path was not supplied, grab a reference to the camera
        if not args.get("input", False):
                print("[INFO] Starting the live stream..")
                vs = cv2.VideoCapture(config.url)
                if config.Thread:
                                cap = thread.ThreadingClass(config.url)
                time.sleep(2.0)

        # otherwise, grab a reference to the video file
        else:
                print("[INFO] Starting the video..")
                vs = cv2.VideoCapture(args["input"])
                if config.Thread:
                                cap = thread.ThreadingClass(args["input"])

        writer = None
        # start the FPS counter
        fps = FPS().start()

        # loop over the frames from the video stream
        while True:
                # read the next frame from the file
                if config.Thread:
                        frame = cap.read()

                else:
                        (grabbed, frame) = vs.read()
                        # if the frame was not grabbed, then we have reached the end of the stream
                        if not grabbed:
                                break

                # resize the frame and then detect people (and only people) in it
                frame = imutils.resize(frame, width=700)
                results = detect_people(frame, net, ln,
                        personIdx=LABELS.index("person"))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(gray,
                                                 scaleFactor=1.1,
                                                 minNeighbors=5,
                                                 minSize=(60, 60),
                                                 flags=cv2.CASCADE_SCALE_IMAGE)
                serious = set()
                abnormal = set()

                # ensure there are *at least* two people detections (required in
                # order to compute our pairwise distance maps)
                if len(results) >= 2:
                        # extract all centroids from the results and compute the
                        # Euclidean distances between all pairs of the centroids
                        centroids = np.array([r[2] for r in results])
                        D = dist.cdist(centroids, centroids, metric="euclidean")

                        # loop over the upper triangular of the distance matrix
                        for i in range(0, D.shape[0]):
                                for j in range(i + 1, D.shape[1]):
                                        # check to see if the distance between any two
                                        # centroid pairs is less than the configured number of pixels
                                        if D[i, j] < config.MIN_DISTANCE:
                                                # update our violation set with the indexes of the centroid pairs
                                                serious.add(i)
                                                serious.add(j)
                        # update our abnormal set if the centroid distance is below max distance limit
                                        if (D[i, j] < config.MAX_DISTANCE) and not serious:
                                                abnormal.add(i)
                                                abnormal.add(j)

                # loop over the results
                for (i, (prob, bbox, centroid)) in enumerate(results):
                        # extract the bounding box and centroid coordinates, then
                        # initialize the color of the annotation
                        (startX, startY, endX, endY) = bbox
                        (cX, cY) = centroid
                        color = (0, 255, 0)

                        # if the index pair exists within the violation/abnormal sets, then update the color
                        if i in serious:
                                color = (0, 0, 255)
                        elif i in abnormal:
                                color = (0, 255, 255) #orange = (0, 165, 255)

                        # draw (1) a bounding box around the person and (2) the
                        # centroid coordinates of the person,
                        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                        cv2.circle(frame, (cX, cY), 5, color, 2)

                # draw some of the parameters
                Safe_Distance = "Safe distance: >{} px".format(config.MAX_DISTANCE)
                cv2.putText(frame, Safe_Distance, (470, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)
                Threshold = "Threshold limit: {}".format(config.Threshold)
                cv2.putText(frame, Threshold, (470, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)

            # draw the total number of social distancing violations on the output frame
                text = "Total serious violations: {}".format(len(serious))
                cv2.putText(frame, text, (10, frame.shape[0] - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)

                text1 = "Total abnormal violations: {}".format(len(abnormal))
                cv2.putText(frame, text1, (10, frame.shape[0] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)

                faces_list=[]
                preds=[]
                df=pd.read_csv("PersonDetails\PersonDetails.csv")
                font = cv2.FONT_HERSHEY_SIMPLEX
                col_names =  ['Id','Name','Date','Time']
                co=['name']
                mobiless=""
                violation = pd.DataFrame(columns = col_names)
                namess=""
                for index, row in df.iterrows():
                        namess+= str(row['Name'])+""
                        mobiless=mobiless+str(row['Pmono'])+','
                aa=""
                bb=""
                violation1 = pd.DataFrame(columns = co)
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                faces=faceCascade1.detectMultiScale(gray, 1.2,5)
                for (x, y, w, h) in faces:
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(225,0,0),2)
                        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
                        print("conf==",conf)
                        if(conf < 50):
                                ts = time.time()
                                date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                                timeStamp = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                                aa=df.loc[df['Id'] == Id]['Name'].values
                                bb=df.loc[df['Id']==Id]['Pmono'].values
                                print(str(bb))
                                aaa=''.join(e for e in aa if e.isalnum())
                                tt=str(Id)+"-"+aa
                                violation.loc[len(violation)] = [Id,aa,date,timeStamp]
                                violation1.loc[len(violation)] = [aa]
                                namess=namess.replace(aaa, " ")
                                mobiless=mobiless.replace(str(bb)," ")
                        else:
                                Id='Unknown'
                                tt=str(Id)
                        if(conf > 75):
                                #import os
                                noOfFile=len(os.listdir("ImagesUnknown"))+1
                        cv2.putText(frame,str(tt),(x,y+h), font, 1,(255,255,255),2)
                        violation=violation.drop_duplicates(subset=['Id'],keep='first')
                        face_frame = frame[y:y+h,x:x+w]
                        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
                        face_frame = cv2.resize(face_frame, (224, 224))
                        face_frame = img_to_array(face_frame)
                        face_frame = np.expand_dims(face_frame, axis=0)
                        face_frame =  preprocess_input(face_frame)
                        faces_list.append(face_frame)
                        if len(faces_list)>0:
                            preds = model.predict(faces_list)
                            print("preds=",preds)
                        for pred in preds:
                            (mask, withoutMask) = pred
                        label = "Mask" if mask > withoutMask else "No Mask"
                        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                        cv2.putText(frame, label, (x, y- 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                 
                        cv2.rectangle(frame, (x, y), (x + w, y + h),color, 2)
                        # Display the resulting frame
                #cv2.imshow('Video', frame)

                

        #------------------------------Alert function----------------------------------#
                if (mask < withoutMask):
                        msg=""
                        mobile2=""
                        config.ALERT=True
                        print("mask score==",mask)
                        print("Without_mask score==",withoutMask)
                        #loc=g.get_loc()
                        if mask<withoutMask:
                                msg="People didn't wear the Mask You Can take action"
                        if str(bb)!="":
                                mobile=str(bb)
                                mobile1=mobile.replace('[','')
                                mobile2=mobile.replace(']','')
                                mobile2=mobile2.replace('[','')
                                msg="You didn't wear the Mask You have to pay fine to Pay Rs.2000/- fine to BBMP"
                                #config.ALERT=False
                                        
                                                                    
                        #cv2.putText(frame, "-ALERT: Violations over limit-", (10, frame.shape[0] - 80),
                         #       cv2.FONT_HERSHEY_COMPLEX, 0.60, (0, 0, 255), 2)
                        if config.ALERT:
                                #g = geocoder.ip('me')
                                url = "https://www.fast2sms.com/dev/bulk"
                                print("msg==",msg)
                                mobile=mobile2
                                print("mobile number==",mobile)
                                print('[INFO] playing sound...')
                                
                                payload = "sender_id=FSTSMS&message="+msg+"&language=english&route=p&numbers="+mobile
                                headers = {'authorization':"vZgIqyn8fmbEGF3AH1eRa679w5SULslhtVCJpizQP0WBjr2OdYhECYvioKy6z2la7nMJbZAFke8NSOLV",'Content-Type': "application/x-www-form-urlencoded",'Cache-Control': "no-cache",}
                                response = requests.request("POST", url, data=payload, headers=headers)
                                print("response==",response.text)
                                playsound("alarm.wav")
                                print('[INFO] playing')
                                config.ALERT = False
                                break
                else:
                        config.ALERT=False
                #break
                        
                        
        #------------------------------------------------------------------------------#
                # check to see if the output frame should be displayed to our screen
                if args["display"] > 0:
                        # show the output frame
                        cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
                        key = cv2.waitKey(1) & 0xFF

                        # if the `q` key was pressed, break from the loop
                        if key == ord("q"):
                                break
            # update the FPS counter
                fps.update()

                # if an output video file path has been supplied and the video
                # writer has not been initialized, do so now
                if args["output"] != "" and writer is None:
                        # initialize our video writer
                        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                        writer = cv2.VideoWriter(args["output"], fourcc, 25,
                                (frame.shape[1], frame.shape[0]), True)

                # if the video writer is not None, write the frame to the output video file
                if writer is not None:
                        writer.write(frame)

        # stop the timer and display FPS information
        fps.stop()
        print("===========================")
        print("[INFO] Elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

        # close any open windows
        cv2.destroyAllWindows()
