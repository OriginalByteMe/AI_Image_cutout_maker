import beam
app = beam.App(
    name="cutout_generator",
    cpu=1,
    gpu="T4",
    memory="16Gi",
    python_version="python3.8",
    python_packages="requirements.txt",
    commands=["apt-get update && apt-get install -y ffmpeg"]
)

app.Trigger.TaskQueue(
    inputs={"images": beam.Types.Json() , "prompt": beam.Types.String()},
    handler="run.py:generate_cutout"
)

# File to store image outputs
app.Output.Dir(path="generated_images", name="cutout_images")
app.Mount.PersistentVolume(path="./uploaded_images", name="images")
app.Mount.PersistentVolume(path="./masks", name="masks")
app.Mount.PersistentVolume(path="./models", name="models")
app.Mount.PersistentVolume(path="./svg", name="svgs")