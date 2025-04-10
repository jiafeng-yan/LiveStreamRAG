import requests
import os
os.environ["HF_ENDPOINT"] = 'https://hf-mirror.com'
os.environ["HF_HOME"] = "E:/data/huggingface"


from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch



device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)

# prompt = "<OD>"

# url = "https://hf-mirror.com/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
# image = Image.open(requests.get(url, stream=True).raw)

# inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

# generated_ids = model.generate(
#     input_ids=inputs["input_ids"],
#     pixel_values=inputs["pixel_values"],
#     max_new_tokens=1024,
#     do_sample=False,
#     num_beams=3
# )
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

# parsed_answer = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))

# print(parsed_answer)


def run_example(task_prompt, text_input=None, image=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))

    print(parsed_answer)

# run_example("<OD>", text_input='Can you describe the image in Chinese?', image=Image.open(r"D:\Codefield\Python\LLM_RAG\data\test_screenshots\mock_comments.png"))
run_example("<OCR>", text_input=None, image=Image.open(r"D:\Codefield\Python\LLM_RAG\data\test_screenshots\mock_comments.png"))