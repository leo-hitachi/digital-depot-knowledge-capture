from google import genai
from PIL import Image
from io import BytesIO

client = genai.Client()

prompt = "My this image look like a magazine cover"

image = Image.open('DSCF5463.JPG')
dress = Image.open('DSCF5463.JPG')

response = client.models.generate_content(
    model="gemini-2.5-flash-image-preview",
    contents=[prompt, image],
)

for part in response.candidates[0].content.parts:
  if part.text is not None:
    print(part.text)
  elif part.inline_data is not None:
    image = Image.open(BytesIO(part.inline_data.data))   
    image.save("generated_image.png")
