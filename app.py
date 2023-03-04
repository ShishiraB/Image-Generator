import tkinter as tk
import customtkinter as ctk

from PIL import ImageTk
from authtoken import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

#application window
app = tk.Tk()
app.geometry("532x632")
app.title("Image Generator")

ctk.set_appearance_mode("light")
text = ctk.CTkEntry(app, height=40, width=512, font=("Roboto", 20), text_color="black", fg_color="white")
text.place(x=10, y=10)
imain = ctk.CTkLabel(app, height=512, width=512)
imain.place(x=10, y=100)
model = "CompVis/stable-diffusion-v1-4"
dev = "cpu"
pipe = StableDiffusionPipeline.from_pretrained(model, revision=None, torch_dtype=torch.float64, use_auth_token=auth_token)
pipe.to(dev)

def gen():
    with autocast(dev):
        image = pipe(text.get(), guidance_scale=8.5)["Sample"][0]
    image.save('output.png')
    img = ImageTk.PhotoImage(image)
    imain.configure(image=img)
trig = ctk.CTkButton(app, height=40, width=120,font=("Roboto", 20), text_color="white", fg_color="orange", command=gen)
trig.configure(text="Create")
trig.place(x=206, y=60)

app.mainloop()