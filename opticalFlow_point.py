import numpy as np
import cv2 as cv
import Utils

#https://docs.opencv.org/4.9.0/d4/dee/tutorial_optical_flow.html
#https://docs.opencv.org/4.9.0/dc/d6b/group__video__track.html

# Parameters for lucas kanade optical flow
lk_params = dict(winSize  = (15, 15),
                maxLevel = 3,
                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

point2D = np.load("data/cdf/Walking 1_h36m_points2D.npy")

#Cámara 1
video_path = 'video/Walking 1.54138969.mp4'
point_2d = point2D[:2, :]

#Cámara 2
#video_path = 'video/Walking 1.55011271.mp4'
#point_2d = point2D[3:5, :]

#Cámara 3
#video_path = 'video/Walking 1.58860488.mp4'
#point_2d = point2D[6:8, :]

#Cámara 4
#video_path = 'video/Walking 1.60457274.mp4'
#point_2d = point2D[9:11, :]
print("point_2d=\n", point_2d)

#video_path = 'bs_output/bs_MOG2_Walking 1.54138969.mp4'
cap = cv.VideoCapture(video_path)

# Take first frame and find corners in it
ret, frame = cap.read()

old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

# Create a mask image for drawing purposes
mask = np.zeros_like(frame)

total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

print("El video tiene {} fotogramas.".format(total_frames))

# out = Utils.create_output_video("seguimiento_flujo_optico.mp4")
chunk_points = np.array_split(point_2d, 111232 // 32, axis=1)
index_frame = 0
frame_points = chunk_points[index_frame]
old_points = np.array([frame_points[0], frame_points[1]]).T.reshape(-1, 1, 2).astype(np.float32)

_, unique_indices = np.unique(old_points, axis=0, return_index=True)
old_points = old_points[unique_indices]
#print("old_points=\n", old_points)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print('No frames grabbed!')
        break
    
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # calculate optical flow
    new_points, status, error = cv.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
    old_gray = gray_frame.copy()
    
    # draw the tracks
    for i, (new, old) in enumerate(zip(new_points, old_points)):
        a, b = new.ravel()
        c, d = old.ravel()
        #mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        #img = cv.add(frame, mask)
    
    first_level = cv.pyrDown(frame)
    second_level = cv.pyrDown(first_level)
    third_level = cv.pyrDown(second_level)
    fourth_level = cv.pyrDown(third_level)
    fifth_level = cv.pyrDown(fourth_level)
    sixth_level = cv.pyrDown(fifth_level)

    cv.imshow('Frame', frame)
    cv.imshow('First level', first_level)
    cv.imshow('Second level', second_level)
    cv.imshow('Third level', third_level)
    #cv.imshow('frame', frame)
    #cv.imshow('frame', frame)
    #cv.imshow('frame', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    old_points = new_points.reshape(-1, 1, 2)

cv.destroyAllWindows()