import numpy as np
import cv2 as cv
import Utils

#https://docs.opencv.org/4.9.0/d4/dee/tutorial_optical_flow.html
#https://docs.opencv.org/4.9.0/dc/d6b/group__video__track.html

# Parameters for lucas kanade optical flow
lk_params = dict(winSize  = (21, 21),
                maxLevel = 5,
                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

#point2D = np.load("data/cdf/Walking 1_h36m_points2D.npy")

#Cámara 1
video_path = 'video/Walking 1.54138969.mp4'
#point_2d = point2D[:2, :]

#Cámara 2
#video_path = 'video/Walking 1.55011271.mp4'
#point_2d = point2D[3:5, :]

#Cámara 3
#video_path = 'video/Walking 1.58860488.mp4'
#point_2d = point2D[6:9, :]

#Cámara 4
#video_path = 'video/Walking 1.60457274.mp4'
#point_2d = point2D[10:12, :]

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
#chunk_points = np.array_split(point_2d, 111232 // 32, axis=1)
index_frame = 0
#frame_points = chunk_points[index_frame]
#print("frame_points=\n", frame_points)
#old_points = np.array([frame_points[0], frame_points[1]]).T.reshape(-1, 1, 2).astype(np.float32)
#print("old_points=\n", old_points)

point = ()
point_selected = False
old_points =  np.array([[0, 0]], dtype = np.float32)

# Create a mask image for drawing purposes
mask = np.zeros_like(frame)
mask[:] = 255

def select_point(event, x, y, flags, params):
    global point, point_selected, old_points
    if event == cv.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_selected = True
        old_points = np.array([[x, y]]).reshape(-1, 1, 2).astype(np.float32)

cv.namedWindow("Frame")
cv.setMouseCallback("Frame", select_point)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print('No frames grabbed!')
        break
    
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    if(point_selected is True):
        cv.circle(frame, point, 5, (0, 0, 255), 2)

        # calculate optical flow
        new_points, status, error = cv.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)

        old_gray = gray_frame.copy()
        
        # draw the tracks
        for i, (new, old) in enumerate(zip(new_points, old_points)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            #img = cv.add(frame, mask)

        old_points = new_points.reshape(-1, 1, 2)
    
    first_level = cv.pyrDown(frame)
    second_level = cv.pyrDown(first_level)
    third_level = cv.pyrDown(second_level)
    fourth_level = cv.pyrDown(third_level)
    fifth_level = cv.pyrDown(fourth_level)

    Utils.draw_title(frame, "Tracking Optical flow")
    Utils.draw_title(mask, "Mask flow", color=(0,0,0))
    tmp1 = cv.hconcat([frame, mask])
    final_img = cv.vconcat([tmp1])

    cv.imshow('Optical flow', final_img)
    cv.imshow('Frame', frame)

    h, w = frame.shape[:2]
    first_level_resized = cv.resize(first_level, (w, h))
    second_level_resized = cv.resize(second_level, (w, h))
    third_level_resized = cv.resize(third_level, (w, h))
    fourth_level_resized = cv.resize(fourth_level, (w, h))
    fifth_level_resized = cv.resize(fifth_level, (w, h))
    Utils.draw_title(first_level_resized, "First level")
    Utils.draw_title(second_level_resized, "Second level")
    Utils.draw_title(third_level_resized, "Third level")
    Utils.draw_title(fourth_level_resized, "Fourth level")
    Utils.draw_title(fifth_level_resized, "Fifth level")

    tmp1 = cv.hconcat([frame, first_level_resized, second_level_resized])
    tmp2 = cv.hconcat([third_level_resized, fourth_level_resized, fifth_level_resized])

    pyramid_levels = cv.vconcat([tmp1, tmp2])
    #cv.imshow('Pyramid level', pyramid_levels)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    
cap.release()
cv.destroyAllWindows()