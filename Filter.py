import numpy as np
import random
from Particle import Particle

class Filter:

    def __init__(self, particle_number, particle_size, img_width, img_height, perturbation_size=20):
        if img_width // 3 <= particle_size[0]:
            raise ValueError("Check the size of particle.")

        if img_height // 3 <= particle_size[1]:
            raise ValueError("Check the size of particle.")

        self.particle_number = particle_number
        self.particle_size = particle_size
        self.img_width = img_width
        self.img_height = img_height
        self.perturbation_size = perturbation_size
        self.particles = []
        self.particles_perturbation = []
        self.particles_prediction = []
        self.total_score = 0
        self.chosen_particles = []

    # Crear partículas
    def initialise_random(self):
        if self.particles_perturbation != []:
            self.particles = self.particles_perturbation
        else:
            self.particles = [
                self.create_random_particle(i)
                for i in range(self.particle_number)
            ]

    def create_random_particle(self, index):
        x_part = random.randint(0, self.img_width - self.particle_size[0])
        y_part = random.randint(0, self.img_height - self.particle_size[1])
        return Particle(x_part, y_part, self.particle_size[0], self.particle_size[1], index)

    def perturbation(self):
        self.particles_perturbation = []

        if self.total_score > 0:
            x_max, y_max = 0, 0
            x_min, y_min = float('inf'), float('inf')
            ids = []
            scores = []

            for particle in self.particles:

                ids.append(particle.get_id())
                scores.append(particle.get_score())

                # Actualizar los valores mínimo y máximo de x y y para perturbaciones gaussianas
                if particle.get_score() > 0:
                    x, y, w, h = particle.get_coordinates()
                    x_max = max(x, x_max)
                    y_max = max(y, y_max)
                    x_min = min(x, x_min)
                    y_min = min(y, y_min)

            # Método de la ruleta para seleccionar partículas
            probabilities = np.array(scores, dtype=int) / int(sum(scores))
            random_particles = np.random.choice(ids, self.particle_number, p=probabilities)

            # Crear partículas perturbadas
            for i, sample in enumerate(random_particles):
                x, y, w, h = self.particles[sample].get_coordinates()

                x = int(np.random.normal(x, self.perturbation_size))
                y = int(np.random.normal(y, self.perturbation_size))

                # Verificar que las perturbaciones no estén fuera del rango de la imagen
                x = max(0, min(x, self.img_width - self.particle_size[0]))
                y = max(0, min(y, self.img_height - self.particle_size[1]))

                particle = Particle(x, y, w, h, i)
                self.particles_perturbation.append(particle)

    def calculate_score(self, mask):
        nb_white_pxls = np.sum(mask == 255)
        score = 0
        total_score = 0

        # Calculating the score for each particle (white pixels)
        for particle in self.particles:
            x, y, w, h = particle.get_coordinates()

            if nb_white_pxls > 0:
                crop = mask[y:y+h, x:x+w]
                score = np.sum(crop == 255)
                total_score += score

            particle.set_score(score)

        self.total_score = total_score


    def choose_particle(self):
        if self.total_score <= 0:
            return
        
        # Encontrar la partícula con el score más alto
        max_particle = max(self.particles, key=lambda p: p.get_score(), default=None)

        if max_particle is not None:
            max_particle.set_chosen(True)
            self.chosen_particles.append(max_particle)

    def prediction(self):
        # Si hay menos de dos partículas seleccionadas, no hacer predicciones
        if len(self.chosen_particles) < 2:
            return
        
        self.particles_prediction = []

         # Obtener las coordenadas de las dos últimas partículas seleccionadas
        x1, y1, w1, h1 = self.chosen_particles[-1].get_coordinates()
        x2, y2, _, _ = self.chosen_particles[-2].get_coordinates()

        # Calcular la dirección de movimiento
        dir_x = x1 + (x1 - x2)
        dir_y = y1 + (y1 - y2)

        # Generar nuevas partículas perturbadas alrededor de la dirección calculada
        self.particles_prediction = [
            Particle(
                int(np.random.normal(dir_x, self.perturbation_size)),
                int(np.random.normal(dir_y, self.perturbation_size)),
                w1, h1, i
            ) for i in range(self.particle_number)
        ]

    def get_particles(self):
        return self.particles

    def get_chosen_particles(self):
        return self.chosen_particles

    def get_particles_perturbation(self):
        return self.particles_perturbation

    def get_particles_prediction(self):
        return self.particles_prediction
