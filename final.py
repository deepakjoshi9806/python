import face_recognition
from sklearn import svm
import os
import cv2

encodings = []
names = []


def face_recognize(dir):
    if dir[-1] != '/' :
        dir += '/'
    train_dir = os.listdir(dir)
    # Loop through each person in the training directory
    for person in train_dir :
        pix = os.listdir(dir + person)
        # Loop through each training image for the current person
        for person_img in pix :
            # Get the face encodings for the face in each image file
            face = face_recognition.load_image_file(
                dir + person + "/" + person_img)
            face_bounding_boxes = face_recognition.face_locations(face)

            # If training image contains exactly one face
            if len(face_bounding_boxes) == 1 :
                face_enc = face_recognition.face_encodings(face)[0]
                # Add face encoding for current image
                # with corresponding label (name) to the training data
                encodings.append(face_enc)
                names.append(person)
            else :
                print(person + "/" + person_img + " can't be used for training")
                # for training each img must contain 1 sample img
    print("training complete\n ")
    clf = svm.SVC(gamma='scale')
    clf.fit(encodings, names)
    video_capture = cv2.VideoCapture(0)
    while True :
        ret, frame = video_capture.read()
        rgb_frame = frame[:, :, : :-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        for top, right, bottom, left in face_locations :
            test_image_enc = face_recognition.face_encodings(rgb_frame[top:bottom, left:right])
            # print([test_image_enc])
            name = clf.predict(test_image_enc)  # issue here
            print(*name)
            # name = name.reshape(-1, 1)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, *name, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    train_dir = "C:/resources/train_dir"
    print("process initiated! \n")
    face_recognize(train_dir)


if __name__ == "__main__" :
    main()