from pilib import pilib as pi

# projeto_pi\\shirt.png
#pi.imread("projeto_pi\\shirt.png")

banana = pi.imread("banana.jpg")
eye = pi.imread("eye.jpg")
shirt = pi.imread("shirt.png")
guy = pi.imread("guy.jpg")

"""
print "2. a)"
print banana
print "2. b)"
print eye
print "2. c)"
print shirt


print "Q3."
print "Numero de canais (grayscale):"
print pi.nchannels(eye)
print pi.nchannels(shirt)
print "Numero de canais (rgb):"
print pi.nchannels(banana)

print "\nQ4."
print "Dimensoes (olho):"
print pi.size(eye)
print "Dimensoes (camisa):"
print pi.size(shirt)
print "Dimensoes (banana):"
print pi.size(banana)

print "\nQ5."
print "Antes:"
print guy
print "\nDepois:"
print pi.rgb2gray(guy)
"""

tim = pi.thresh(shirt, 190)
pi.imshow(tim)