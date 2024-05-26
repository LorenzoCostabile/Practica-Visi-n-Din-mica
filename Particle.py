class Particle:

    def __init__(self, x, y, w, h, id):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.score = 0
        self.chosen = False
        self.id = id

    def get_coordinates(self):
        return self.x, self.y, self.w, self.h

    def get_score(self):
        return self.score

    def get_chosen(self):
        return self.chosen

    def get_id(self):
        return self.id

    def set_score(self, score):
        self.score = score

    def set_chosen(self, bool):
        self.chosen = bool
