# Liveness Detection

There are multiple ways to we can fix the spoofing in face recognition

1. Eye Blink Detection
2. Move a face to fix it on bounding box
3. Head Movement

I have used the dlib --> frontal face - 68 landmarks

any issues in installing dlib use this:
pip install https://pypi.python.org/packages/da/06/bd3e241c4eb0a662914b3b4875fc52dd176a9db0d4a2c915ac2ad8800e9e/dlib-19.7.0-cp36-cp36m-win_amd64.whl#md5=b7330a5b2d46420343fbed5df69e6a3f

please download the file from below path and extract .dat file and place it in folder face_landmark_dat
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
