from django.db import models

class Frase (models.Model):
    text = models.TextField()

# methods
def get_text(self):
    return self.text