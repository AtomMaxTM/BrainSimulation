import tkinter as tk
import random
import time
# We don't need something really advance and fast
# so I believe, we can just use tkinter package
# for that

# Creating scene
root = tk.Tk()

# Creating canvas, whatever that means
canvas = tk.Canvas(root, width=600, height=400)
canvas.pack()

# Here we write, what to print, on the screen
# canvas.create_rectangle(50, 50, 150, 150, fill="#800080")
#------------------------------------#
# I want to print just some layers with neurons, and make them centered to the center
# I want to have limitation at first by 6 rows and 5 columns for
# development was easy and doesn't move us away from creating brain
# to just details

# I want to have layers and print there neurons as squares of different colors
# more sensetive neurons are gonna be more red
# less sensetive are green
# Then I would like to draw arrows between them
# the rest are pretty simple and straighforward
# and I don't want to write about it.
# such things as changing color due to firing neurons
#------------------------------------#

# example for input
test_list = [5,3,3,5]
sumi = sum(test_list)

# now I need to calculate, how to place them
# It's good to find max number of neurons in layer and place them first

preoddwidest = [x for x in test_list if x % 2 == 1]
if len(preoddwidest) != 0:
    oddwidest = max(preoddwidest)
else:
    oddwidest = 0
oddcords = [(70+x*170) for x in range(oddwidest)]

preevenwidest = [x for x in test_list if x % 2 == 0]
if len(preevenwidest) != 0:
    evenwidest = max(preevenwidest)
else:
    evenwidest = 0
evencords = [(70+x*170) for x in range(evenwidest)]

if oddwidest > evenwidest:
    widest = oddwidest
    cords = oddcords
else:
    widest = evenwidest
    cords = evencords

ram = 70
c = 0


canvas.config(width=70+170*widest, height=100+(len(test_list)*100+(len(test_list)+1)*50))

list1 = []

def draw_rectangle(muly,mulx):
    global counter, cords, list1
    list1.append(canvas.create_rectangle(cords[muly], 70+170*mulx, cords[muly]+100, 170*(mulx+1), fill="#800080"))
    print("draw_rectangle called")

majorcounter = 1
for x in range(len(test_list)):
    if (test_list[x] - widest) % 2 == 0:
        print("inside range x",x)
        minimum = (widest-test_list[x]) // 2
        maximum = widest - minimum
        counter = minimum
        root.update()
        while counter < maximum:
            draw_rectangle(counter,x)
            majorcounter += 1
            counter += 1





root.mainloop()