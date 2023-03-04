import tkinter as tk
import customtkinter as ctk

from PIL import ImageTk
from auth.authtoken import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

#application window
app = tk.Tk()
app.geometry("800x600")
app.title("Image Generator")
ctk.set_appearance_mode("light")
text = ctk.CTkEntry(app, height=40, width=780, font=("Roboto", 20), text_color="black", fg_color="white")
text.place(x=10, y=10)

#Image Generation
def gen():
    with autocast(dev):
        image = pipe(text.get(), guidance_scale=8.5).images[0]
    image.save('output.png')
    img = ImageTk.PhotoImage(image)
    imain.configure(image=img)

#Button
trig = ctk.CTkButton(app, height=40, width=350,font=("Roboto", 20), text_color="white", fg_color="orange", command=gen)
trig.configure(text="Create")
trig.place(x=206, y=60)
imain = ctk.CTkLabel(app, height=500, width=500)
imain.place(x=50, y=100)

model = "CompVis/stable-diffusion-v1-4"
dev = "cpu"
pipe = StableDiffusionPipeline.from_pretrained(model, revision=None, torch_dtype=torch.float64, use_auth_token=auth_token)
pipe.to(dev)

app.mainloop()