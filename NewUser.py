import cv2
import os
import sqlite3

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# The funtion update name and id to database
def insertOrUpdate(id, name):
    conn = sqlite3.connect("FaceBaseNew.db")
    cursor = conn.execute('SELECT * FROM People WHERE ID='+str(id))
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1
        break

    if isRecordExist == 1:
        cmd = "UPDATE people SET Name=' "+str(name)+" ' WHERE ID="+str(id)
    else:
        cmd = "INSERT INTO people(ID, Name) Values("+str(id)+",' "+str(name)+" ' )"

    conn.execute(cmd)
    conn.commit()
    conn.close()

def create_new_user():
    cam = cv2.VideoCapture(0)
    id = input('Enter staff code: ')
    name = input('Enter staff name: ')
    print("Start to take photos ,Enter q to exit !")

    insertOrUpdate(id, name)

    sampleNum = 0

    while(True):
        ret, img = cam.read()

        img = cv2.flip(img, 1)

        centerH = img.shape[0] // 2
        centerW = img.shape[1] // 2
        sizeboxW = 300
        sizeboxH = 400
        cv2.rectangle(img, (centerW - sizeboxW // 2, centerH - sizeboxH // 2),
                      (centerW + sizeboxW // 2, centerH + sizeboxH // 2), (255, 255, 255), 5)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        parent_dir = "dataSet"
        img_dir = id
        path = os.path.join(parent_dir, img_dir)
        if not os.path.isdir(path):
            os.mkdir(path)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sampleNum = sampleNum + 1
            cv2.imwrite(parent_dir+'/'+id +"/User." + id + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])

        cv2.imshow('frame', img)
        # Check if enter q or over 100 photos
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        elif sampleNum > 100:
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    create_new_user()