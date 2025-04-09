from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.pydantic_v1 import BaseModel, Field
import time
import json
import os
import pandas as pd
import logging
from tqdm import tqdm
import traceback
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(filename='final_llama_generation.log', level=logging.INFO)


class FinalDescription(BaseModel):
    description: str = Field(description="description")


llm = ChatGroq(model="llama3-70b-8192", temperature=0.3)
template = '''
You're large language model that cross checks description and some of the objects with their probabilities provided by object detection model
We need to filter out objects whose probabilities are lower than 0.02 and rewrite the image description 
to only include objects with higher probabilities.

Please rewrite the description, ensuring that you only include the objects that have a probability greater than 0,
 and remove all objects that fall below this threshold. Be sure to provide a clear, natural-sounding description with only the remaining objects.
'''

system_message = SystemMessagePromptTemplate.from_template(template)
human_message = HumanMessagePromptTemplate.from_template("Here is the initial description of the image: {description}"
                                                         "The detected objects and their probabilities are: {objects_and_probs}")

prompt_final_llama = ChatPromptTemplate.from_messages([system_message, human_message])
model = prompt_final_llama | llm.with_structured_output(FinalDescription)


request_counter = 0
last_reset_time = time.time()


def invoke_with_rate_limit_per_minute(description, objects_and_probs, requests_per_minute=29):
    global request_counter, last_reset_time

    current_time = time.time()
    if current_time - last_reset_time >= 60:
        request_counter = 0
        last_reset_time = current_time

    if request_counter >= requests_per_minute:
        wait_time = 60 - (current_time - last_reset_time)
        print(f"Rate limit reached. Please wait {wait_time:.2f} seconds.")
        time.sleep(wait_time)
        request_counter = 0
        last_reset_time = time.time()

    # Now make the request
    response = model.invoke({"description": description, "objects_and_probs": objects_and_probs}).description

    # Increment the request counter after the request
    request_counter += 1

    return response


root = r"C:\Users\Gram\Desktop\NULP\uav_img_cap"

path_to_final_llama = os.path.join(root, "bin/inference_final_llama80b_3.json")
path_to_first_llama = os.path.join(root, "bin/inference_llama_first_80b.json")
path_to_object_detection = os.path.join(root, r"C:\Users\Gram\Desktop\NULP\uav_img_cap\notebooks\object_detection_max_probs_by_objects.csv")


if not os.path.exists(path_to_final_llama):
    with open(path_to_final_llama, "w") as f:
        json.dump([], f)

with open(path_to_first_llama, "r") as f:
    first_llama_inference = json.loads(f.read())

with open(path_to_final_llama, "r") as f:
    final_llama_inference = json.loads(f.read())

object_detection_df = pd.read_csv(path_to_object_detection)

first_llama_df = pd.DataFrame(first_llama_inference)
checked_images = [] if len(final_llama_inference) == 0 else pd.DataFrame(final_llama_inference).image.unique().tolist()
logger.info(f"amount of generated descriptions {len(checked_images)}")
merged_df = first_llama_df.merge(object_detection_df, on="image")


logger.info("started generating descriptions")
for i, r in tqdm(merged_df[~merged_df.image.isin(checked_images)].iterrows(),
                 total=(~merged_df.image.isin(checked_images)).sum()):
    try:
        final_llama_inference.append({"image": r["image"],
                                      "final_description": invoke_with_rate_limit_per_minute(r["description"], r["max_probs_by_objects"])})
    except Exception as e:
        logger.info(f"got exception {traceback.format_exc()}")
        raise e
    finally:
        with open(path_to_final_llama, "w") as f:
            json.dump(final_llama_inference, f, indent=4)
        logger.info(f"finished with {len(final_llama_inference)} data points processed")
