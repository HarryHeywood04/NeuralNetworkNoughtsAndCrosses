import random
from tkinter import *
from tensorflow import keras
import numpy as np

# SETTINGS
AI_MODEL = 'nexus'  # Change which model to use here

playerTurn = True
state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
buttons = [0, 0, 0, 0, 0, 0, 0, 0, 0]
model = keras.models.load_model('models/' + AI_MODEL + '.h5')
print("LOADED " + AI_MODEL.upper() + ":")
model.summary()


def reset():
    global state, buttons, playerTurn
    state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    playerTurn = True
    win_text.config(text="")
    for i in range(9):
        buttons[i].destroy()
        buttons[i] = Button(
            height=4, width=8,
            font=("Helvetica", "20"),
            command=lambda p=i: place(p))
        buttons[i].grid(row=int(i / 3), column=int(i % 3))


def win(player):
    global win_text
    if player == 1:
        win_text.config(text="Player wins")
    elif player == 2:
        win_text.config(text="AI wins")
    else:
        win_text.config(text="It's a draw")


def winCheck():
    global state
    if state[0] == state[1] == state[2] and state[0] != 0:
        return state[0]
    if state[3] == state[4] == state[5] and state[3] != 0:
        return state[3]
    if state[6] == state[7] == state[8] and state[6] != 0:
        return state[6]
    if state[0] == state[3] == state[6] and state[0] != 0:
        return state[0]
    if state[1] == state[4] == state[7] and state[1] != 0:
        return state[1]
    if state[2] == state[5] == state[8] and state[2] != 0:
        return state[2]
    if state[0] == state[4] == state[8] and state[0] != 0:
        return state[0]
    if state[2] == state[4] == state[6] and state[2] != 0:
        return state[2]
    if state[0] != 0 and state[1] != 0 and state[2] != 0 and state[3] != 0 and state[4] != 0 and state[5] != 0 and \
            state[6] != 0 and state[7] != 0 and state[8] != 0:
        return 0
    return -1


def place(slot):
    global playerTurn
    if playerTurn and state[slot] == 0:
        state[slot] = 1
        buttons[slot].config(text="X")
        playerTurn = False
        player = winCheck()
        if player == -1:
            runAI()
        else:
            win(player)


def placeAI(slot):
    global playerTurn
    state[slot] = 2
    buttons[slot].config(text="O")
    player = winCheck()
    if player == -1:
        playerTurn = True
    else:
        win(player)


def runAI():
    move = np.argmax(model.predict([state]))
    if state[move] == 0:
        placeAI(move)
    else:
        print("AI chose a filled tile, randomly picking new tile.")
        slot = random.randint(0, 8)
        while state[slot] != 0:
            slot = random.randint(0, 8)
        placeAI(slot)


root = Tk()
root.title("Noughts and crosses")

for i in range(9):
    buttons[i] = Button(
        height=4, width=8,
        font=("Helvetica", "20"),
        command=lambda p=i: place(p))
    buttons[i].grid(row=int(i / 3), column=int(i % 3))
Button(height=2, width=8, font=("Helvetica", "20"), text="reset", command=reset).grid(row=3, column=2)
win_text = Label(height=2, width=16, font=("Helvetica", "20"), text="")
win_text.grid(row=3, column=0, columnspan=2)

mainloop()
