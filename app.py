import beam

app = beam.App(
    name="cutout_generator",
    cpu=1,
    gpu="T4",
    memory="16Gi",
    python_version="python3.8",
    python_packages="requirements.txt",
    commands=["apt-get update && apt-get install -y ffmpeg"],
)

app.Trigger.TaskQueue(
    inputs={
        "image": beam.Types.Image(raw=False),
        "name": beam.Types.String(),
        "prompt": beam.Types.String(),
    },
    handler="run.py:generate_cutout",
)
app.Output.Dir(path="generated_images", name="images")
app.Mount.PersistentVolume(path="./models", name="models")
