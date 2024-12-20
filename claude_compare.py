import anthropic
import argparse
import base64

parser = argparse.ArgumentParser()
parser.add_argument("--image_1", type=str, default = r'/root/photo-background-generation/res/results-sdxl-inpaint-bg-mask-ci--checkpoint_expmts-83,89 final conclusion/checkpoint-89000/1_vaseline.png',required=False)
parser.add_argument("--image_2", type=str, default = r'/root/photo-background-generation/res/sdxl-inpaint-4ch/checkpoint-checkpoint-81730/1_vaseline.png',required=False)
parser.add_argument("--image_3", type=str, default = r'/root/photo-background-generation/res/phot_results/1_vaseline.png',required=False)
args = parser.parse_args()

img_1_path = args.image_1
img_2_path = args.image_2
img_3_path = args.image_3


def file_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        # Read the binary data and encode to base64
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    

img_1_enc = file_to_base64(img_1_path)
img_2_enc = file_to_base64(img_2_path)
img_3_enc = file_to_base64(img_3_path)

client = anthropic.Anthropic(api_key="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")
response = client.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=1000,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "<image_1>"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_1_enc
                    }
                },
                {
                    "type": "text",
                    "text": "</image_1>\n----------\n<image_2>"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_2_enc
                    }
                },
                {
                    "type": "text",
                    "text": "</image_2>\n----------\n<image_3>"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_3_enc
                    }
                },
                {
                    "type": "text",
                    "text": """</image_3>\
                         
                    Compare the three images and order them from best to worst in following categories: 
                    1. Realism of the background of the image 
                    2. Quality and texture of the overall image 
                    3. lighting of the foreground object should match the lighting of the background
                    4. landing of the foreground object. (foreground object should not appear to be floating in air)
                    5. distortion of the background

                    
                    Output in following format: 
                    <category 1>
                    ...
                    </category 1>
                    <category 2>
                    ...
                    </category 2>
                    <category 3>
                    ...
                    </category 3>
                    <category 4>
                    ...
                    </category 4> 
                    <category 5>
                    ...
                    </category 5>                   
                    ...

                    """
                }
            ]
        }
    ]
)

print(response.content[0].text)