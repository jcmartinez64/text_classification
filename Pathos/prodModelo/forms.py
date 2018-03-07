from django import forms

class FraseForm(forms.Form):
    frase = forms.CharField(label='', max_length=60)
