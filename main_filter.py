import cv2 as cv

from Filter import Filter
import BackgroundSubtraction
import Utils

# Global variables
initial_img = None
particle_number = 25
particle_size = [100, 250]

def filter_initialisation(img_shape):
    global particle_number
    global particle_size

    return Filter(particle_number, particle_size, img_shape[1], img_shape[0])


def draw_selection_rectangles_title(img, particles, color_particles = (255, 0, 0), text = None, color=(255, 255, 255)):
    for particle in particles:
        x, y, w, h = particle.get_coordinates()

        if particle.get_score() > 0:
            cv.rectangle(img, (x, y), (x + w, y + h), color_particles, 2)
 

    if text is not None:
        Utils.draw_title(img, text, color)

def get_chosen_particle(particles):
    chosen_particle = None
    for particle in particles:
        if particle.get_chosen():
            chosen_particle = particle
            break
    return chosen_particle

def draw_chosen_rectangle_title(img, particles, color_particles = (255, 0, 0), text = None, color=(255, 255, 255)):
    chosen_particle = get_chosen_particle(particles)

    if(chosen_particle):
        x, y, w, h = chosen_particle.get_coordinates()
        cv.rectangle(img, (x, y), (x + w, y + h), color_particles, 2)

    if text is not None:
        Utils.draw_title(img, text, color)

def draw_images(filter, original_img, mask, i):
    global initial_img

    mask_rgb = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
    all_squares_img = mask_rgb.copy()
    selection_img = mask_rgb.copy()
    chosen_img = mask_rgb.copy()
    prediction_img = mask_rgb.copy()

    draw_selection_rectangles_title(selection_img, filter.get_particles(), color_particles=(102, 102, 202), text="Selection")
    draw_chosen_rectangle_title(chosen_img, filter.get_particles(), color_particles=(0, 0, 255), text="Estimation")
    Utils.draw_rectangles_title(prediction_img, filter.get_particles_prediction(), color_particles=(0, 255, 0), text="Prediction")

    if i == 0:
        initial_img = all_squares_img.copy()
        Utils.draw_rectangles_title(initial_img, filter.get_particles(), color_particles=(0, 0, 0), text="Initialisation", color=(0,0,0))

    Utils.draw_title(original_img, "Original")
    Utils.draw_title(mask_rgb, "Mask")

    tmp1 = cv.hconcat([original_img, mask_rgb, initial_img])
    tmp2 = cv.hconcat([chosen_img, selection_img, prediction_img])
    final_img = cv.vconcat([tmp1, tmp2])

    cv.imshow('Particle filter', final_img)

def main():
    video_path = "video/Walking 1.54138969.mp4"
    index_frame = 0
    cap = cv.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # El primer frame de la máscara con MOG2 es blanco
        mask = BackgroundSubtraction.bs_MOG2(frame)
        #mask = BackgroundSubtraction.bs_KNN(frame)
        #mask = BackgroundSubtraction.bs_GMG(frame)

        if(index_frame == 0):
            particle_filter = filter_initialisation(frame.shape)

        particle_filter.initialise_random()

        particle_filter.calculate_score(mask)
        particle_filter.choose_particle()
        particle_filter.perturbation()
        particle_filter.prediction()

        draw_images(particle_filter, frame, mask, index_frame)
        index_frame += 1

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
