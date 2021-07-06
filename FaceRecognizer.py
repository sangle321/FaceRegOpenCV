import cv2
import sqlite3

# Get info of user from id
def getProfile(id):
    conn = sqlite3.connect("FaceBaseNew.db")
    cursor = conn.execute("SELECT * FROM People WHERE ID="+str(id))
    records = cursor.fetchall()[0]
    conn.close()
    return records

def face_recognize():
    # Init the face detector
    faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Init face reccognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('recognizer/trainner.yml')

    id = 0
    #set text style
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    fontcolor = (0, 255, 0)
    fontcolor1 = (0, 0, 255)

    # Init camera
    cam = cv2.VideoCapture(0)

    while(True):
        ret, img = cam.read()

        img = cv2.flip(img, 1)

        # Draw a rectangular frame to position the area where the user is to be inserted
        centerH = img.shape[0] // 2
        centerW = img.shape[1] // 2
        sizeboxW = 300
        sizeboxH = 400
        cv2.rectangle(img, (centerW - sizeboxW // 2, centerH - sizeboxH // 2),
                      (centerW + sizeboxW // 2, centerH + sizeboxH // 2), (255, 255, 255), 5)

        # Convert image to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceDetect.detectMultiScale(gray, 1.3, 5)

        # Loop through the received faces to reveal information
        for(x, y, w, h) in faces:
            # Draw the rectangle around face
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Face reccognize and return 2 parameter id(ployee code): , distane()
            id, dist = recognizer.predict(gray[y:y+h, x:x+w])

            profile = None

            if (dist < 50):
                profile = getProfile(id)

            # Show info name or unknown if not found
            if(profile != None):
                cv2.putText(img, "Name: " + str(profile[1]), (x, y+h+30), fontface, fontscale, fontcolor, 2)
            else:
                cv2.putText(img, "Name: Unknown", (x, y + h + 30), fontface, fontscale, fontcolor1, 2)

        cv2.imshow('Face', img)
        # Enter q to exit
        if cv2.waitKey(1) == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    face_recognize()
