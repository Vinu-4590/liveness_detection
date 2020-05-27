import cv2

def recognize_faces_in_cam():
    # create cv2 window and initialize VC object
    cv2.namedWindow("Face Recognizer")
    vc = cv2.VideoCapture(0)

    # define necessary variables
    fixed_x1, fixed_y1, fixed_x2, fixed_y2 = 100, 150, 250, 300
    frame_counter = 0
    frame_counter_max = 100
    result = False

    # initialize haarscade frontal face detector
    font = cv2.FONT_HERSHEY_SIMPLEX
    haar_detector = cv2.CascadeClassifier('face_landmark_dat/haarcascade_frontalface_default.xml')

    # looping through the frames in live stream
    while vc.isOpened():

        # read the frame by frame
        _, frame = vc.read()
        img = frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar_detector.detectMultiScale(gray, 1.3, 5)

        # loop through all the faces detected
        for (x, y, w, h) in faces:
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

        # draw the rectangle to fix the face
        img = cv2.rectangle(frame, (fixed_x1, fixed_y1), (fixed_x2, fixed_y2), (255, 255, 255), 2)

        # calculating the pixel difference b/w face detection bounding box & fixed bounding box
        var_pixel = abs(fixed_x1 - x1) + abs(fixed_y1 - y1) + abs(fixed_x2 - x2) + abs(fixed_y2 - y2)
        if var_pixel <= 80:
            result = True

        key = cv2.waitKey(100)

        # show the image
        cv2.imshow("Face Recognizer", img)

        # condition to exit vc2 capture
        if key == ord('q') or frame_counter > frame_counter_max or var_pixel <= 60:  # exit on q
            break

        frame_counter += 1

    # a bit of clean-up
    vc.release()
    cv2.destroyAllWindows()

    return result

print(recognize_faces_in_cam())