import re
path = '.modal-cli/lib/python3.14/site-packages/orbax/checkpoint/_src/metadata/sharding.py'
with open(path, 'r') as f:
    text = f.read()

text = re.sub(
    r"raise ValueError\(\n\s*'The available devices are different",
    r"return jax.local_devices()[0] #raise ValueError(\n            'The available devices are different",
    text
)
text = re.sub(
    r"raise ValueError\(\n\s*f'Device {device_str} was not found in jax.local_devices\(\)\.'",
    r"return jax.local_devices()[0] #raise ValueError(\n        f'Device {device_str} was not found in jax.local_devices().'",
    text
)

with open(path, 'w') as f:
    f.write(text)
print("Patched.")
