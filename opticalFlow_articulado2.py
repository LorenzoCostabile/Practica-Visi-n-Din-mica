import numpy as np
import cv2 as cv
import Utils

lk_params = dict(winSize=(21, 21),
                 maxLevel=8,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

known_x, known_y = [], []
estimated_x, estimated_y = [], []

video_path = 'video/Walking 1.54138969.mp4'
cap = cv.VideoCapture(video_path)

point2D = np.load("data/cdf/Walking 1_h36m_points2D.npy")

out = Utils.create_output_video("seguimiento_flujo_optico.mp4")

point_2d = point2D[:2, :]
chunk_points = np.array_split(point_2d, 111232 // 32, axis=1)
index_frame = 0

# Take first frame and find corners in it
ret, old_frame = cap.read()
gray_prev = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
frame_points = chunk_points[index_frame]
prev_points = np.array([frame_points[0], frame_points[1]]).T.reshape(-1, 1, 2).astype(np.float32)

index_frame += 1

trajectory_len = 10
detect_interval = 5
trajectories = [[(frame_points[0][i], frame_points[1][i])] for i in range(len(frame_points[0]))]
frame_idx = 0

while cap.isOpened() and index_frame < len(chunk_points):
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    img = frame.copy()

    # Draw the 2D points on the frame (real positions)
    frame_points = chunk_points[index_frame]

    for j in range(len(frame_points[0])):
        x, y = int(frame_points[0][j]), int(frame_points[1][j])
        cv.circle(img, (x, y), 5, (0, 255, 0), -1)  # green
        #print("frame_points=\n", frame_points)
        known_x.append(x)
        known_y.append(y)

    print("trajectories=\n", trajectories)
    if len(trajectories) > 0:
        img0, img1 = gray_prev, gray
        p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
        p1, st, err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, st, err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1

        print("p1=\n", p1)
        print("trajectories=\n", len(trajectories))
        new_trajectories = []

        for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            trajectory.append((x, y))
            if len(trajectory) > trajectory_len:
                del trajectory[0]
            new_trajectories.append(trajectory)
            cv.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

        trajectories = new_trajectories

        # Draw all the trajectories
        cv.polylines(img, [np.int32(trajectory) for trajectory in trajectories], False, (255, 0, 0))

    gray_prev = gray.copy()
    out.write(img)
    cv.imshow('Frame', img)

    index_frame += 1

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()
