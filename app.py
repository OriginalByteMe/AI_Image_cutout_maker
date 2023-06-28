import beam

app = beam.App(
    name="cutout_generator",
    cpu=1,
    gpu="T4",
    memory="32Gi",
    python_version="python3.8",
    python_packages="requirements.txt",
    commands=["apt-get update && apt-get install -y ffmpeg"],
)

app.Mount.PersistentVolume(path="./uploaded_images", name="images")
app.Mount.PersistentVolume(path="./masks", name="masks")
app.Mount.PersistentVolume(path="./cutouts", name="cutouts")
app.Mount.PersistentVolume(path="./models", name="models")