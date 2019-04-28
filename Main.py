
from keras.models import load_model
from keras.preprocessing import image as img
import tkinter as tk
from tkinter import Tk, Frame, StringVar, Label, Button, Entry, messagebox, filedialog, END
from tkinter.ttk import Treeview, Separator, Scrollbar
import os
import pandas as pd
import numpy as np


# gets the name of the classification for an image the model predicted
# parameter1 - the results of the image's prediction
def get_prediction(image_prediction):
    classifications = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    for i in range(len(image_prediction[0])):
        if not image_prediction[0][i] == 0:
            return classifications[i]
    return None


# adds an image's prediction's result to the total result data frame
# parameter1 - the results of the model's predictions
# parameter2 - a list with the result of the image's prediction
def add_to_result(result, new_classification):
    last_row_number = result.index.max()
    if not np.isnan(last_row_number):
        result.loc[last_row_number + 1] = new_classification
    else:
        result.loc[0] = new_classification


# saves the prediction's results as a csv file
# parameter1 - the result of the prediction
def save_results(result):
    file = filedialog.asksaveasfile(mode='w', defaultextension=".csv")
    if file is None:  # asksaveasfile return `None` if dialog closed with "cancel".
        return
    result.to_csv(file, index=None)


# displays the result of the prediction by the model to the user
# parameter1 - the result of the prediction
def show_results(result):
    frame_height = result_frame.winfo_height()
    tree = Treeview(result_frame, columns=(1, 2), height=frame_height, show="headings")
    tree.pack(side='left')
    # tree.pack(side='left', fill='both', expand=True)

    tree.heading(1, text="Image Name")
    tree.heading(2, text="Classification Prediction")
    tree.column(1)
    tree.column(2)

    scroll = Scrollbar(result_frame, orient="vertical", command=tree.yview)
    scroll.pack(side='right', fill='y')
    # scroll.pack(side='left', fill='y')
    tree.configure(yscrollcommand=scroll.set)

    for index, row in result.iterrows():
        tree.insert("", END, values=(row[0], row[1]))
    save_button = Button(result_frame, text="Save Results", command=lambda: save_results(result))
    save_button.pack(side='right')


# using given model to predict given path of pictures' classifications
# parameter1 - a given model
# parameter2 - a given path of pictures
def predict(model, path):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    result = pd.DataFrame(columns=['image_name', 'classification'])
    for folder_name in [i for i in os.listdir(path)]:
        for image_name in [i for i in os.listdir(path + '\\' + folder_name)]:
            try:
                image = img.load_img(path + '\\' + folder_name + '\\' + image_name, target_size=(128, 128))
                image = img.img_to_array(image)
                image = np.expand_dims(image, axis=0)
                image_prediction = model.predict(image)
                image_prediction[0] = np.around(image_prediction[0], decimals=3)
                classification_prediction = get_prediction(image_prediction)
                add_to_result(result, [image_name, classification_prediction])
            except:
                print("The image: " + image_name + " is invalid!")
    return result


# removes the results of the prediction from display
def clear(frame):
    frame.destroy()
    result_frame = Frame(root)
    result_frame.pack(side='bottom', fill='both', expand=True)


# gets the chosen path
# parameter1 - the variable in which the chosen path is to be inserted to
# parameter2 - a char that represents whether the path is of the model (m) or of the flowers' directory (f)
def get_path(actual_path, path_kind):
    try:
        path = None
        if path_kind == 'm':
            path = filedialog.askopenfile(parent=root, title='Choose the model\'s file').name
        if path_kind == 'f':
            path = filedialog.askdirectory(parent=root, title='Choose the flower\'s images directory')
        if path is not None:
            actual_path.set(path)
        else:
            messagebox.showerror("Error", "Couldn't get the path!")
    except:
        messagebox.showerror("Error", "Couldn't get the path!")


# checks whether the paths inserted are valid and if so, predicts the images' classifications using the given model
# parameter1 - the path to the model's file
# parameter2 - the path to the flowers' images folder
def check_and_predict(path_to_model, path_to_flowers_folder):
    is_ok = True

    # checks if the path for the flowers' folder is a folder
    if not os.path.isdir(path_to_flowers_folder.get()):
        is_ok = False
        messagebox.showerror("Invalid Path", "The path inserted for the images' folder is not a folder!")

    # checks if the path for the model is a not a directory
    if os.path.isdir(path_to_model.get()):
        is_ok = False
        messagebox.showerror("Invalid Path", "The path inserted for the model's file is a folder!")

    # checks if the paths are not empty
    if path_to_model.get() == False or path_to_flowers_folder.get() == False:
        is_ok = False
        messagebox.showerror("Invalid Path", "One or more of the paths inserted is empty!")

    if is_ok:
        model = None
        try:
            model = load_model(path_to_model.get())
        except:
            messagebox.showerror("Error", "Couldn't load the model!")
        if model is not None:
            result = predict(model, path_to_flowers_folder.get())
            show_results(result)


root = Tk()
root.title('Flower Classification Interface')
root.geometry('500x500')
path_to_model = StringVar()
path_to_flowers_folder = StringVar()

# creates the frames and separator
button_frame = Frame(root)
separator = Separator(root, orient='horizontal')
result_frame = Frame(root)
button_frame.pack(side='top', fill='both', expand=True)
separator.pack(side='top', fill='x')
# separator.pack(side='top', fill='x', expand=True)
result_frame.pack(side='bottom', fill='both', expand=True)

# init_gui(path_to_model, path_to_flowers_folder)

model_label = Label(button_frame, text="Insert the path for the model:")
model_label.grid(column=0, row=1, padx=3, pady=3)
model_fill = Entry(button_frame, width=40, textvariable=path_to_model)
model_fill.grid(column=1, row=1, padx=3, pady=3)
model_button = Button(button_frame, text="Browse", command=lambda: get_path(path_to_model, 'm'))
model_button.grid(column=2, row=1, padx=3, pady=3)
image_label = Label(button_frame, text="Insert the path for images' folder:")
image_label.grid(column=0, row=2, padx=3, pady=3)
image_fill = Entry(button_frame, width=40, textvariable=path_to_flowers_folder)
image_fill.grid(column=1, row=2, padx=3, pady=3)
image_button = Button(button_frame, text="Browse", command=lambda: get_path(path_to_flowers_folder, 'f'))
image_button.grid(column=2, row=2, padx=3, pady=3)
predict_button = Button(button_frame, text="Predict", command=lambda: check_and_predict(path_to_model, path_to_flowers_folder))
predict_button.grid(column=0, row=3, padx=3, pady=10)
predict_button = tk.Button(button_frame, text="Clear", command=clear(result_frame))
predict_button.grid(column=1, row=3, padx=3, pady=10)

root.mainloop()
