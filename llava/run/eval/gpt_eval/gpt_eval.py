import json
import re
import concurrent.futures
import threading
from openai import OpenAI
from tqdm import tqdm
import base64

# client = OpenAI(
#     api_key="sk-9cce8513b947408c8a3477141d869e21",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# def chat(image, prompt):
#     completion = client.chat.completions.create(
#         model="qwen-vl-plus",
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": f"data:image/png;base64,{image}"
#                         },
#                     },
#                     {"type": "text", "text": prompt},
#                 ],
#             }
#         ],
#     )

#     return completion.choices[0].message.content

client_qwen = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

models_qwen = client_qwen.models.list()
model_qwen = models_qwen.data[0].id

def call_model(model, client, messages, max_tokens=4096, temperature=0.01, top_p=0.1, timeout=2000):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
    )
    return completion.choices[0].message.content


def chat(model, image, prompt):
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "text", 
    #                 "text": prompt
    #             },
    #             {
    #                 "type": "image_url",
    #                 "image": image, 
    #                 "min_pixels": 1280*28*28,
    #                 "max_pixels": 16384*28*28,
    #             },
    #         ],
    #     }
    # ]
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image}"
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


    output = call_model(model, client_qwen, messages, max_tokens=8092, temperature=1, top_p=0.001, timeout=2000)
    return output


def generate_des_prompt(conv):
    prompt = f"""
    Here are a description(including summary and caption) of an image, a question about the image, a reasoning process, and a correct answer: {conv}.
    Now I encounter a trouble of not getting the image, so I am going to answer the question based only on the provided caption. 
    I need your assistance to enhance the original image caption so that it contains all the necessary details of the image for me to answer the question correctly. 
    Ensure your enhanced caption is clear, well-structured, and directly supports answering the question. 
    Output only the enhanced caption with no additional texts.
    """
    return prompt

# ==================== 1) 收集prompts并建立 lines_tracker ==================== #
def collect_prompts_and_build_tracker(input_file):
    """
    读取 input_file, 解析每条JSON, 收集对每轮需要调用API的 prompt。
    同时，构建 lines_tracker: 对每个 line_index 保存处理过程中需要的 merged_conversations 等。

    返回:
      prompts_info: [ {line_index, round_index, prompt, ...}, ... ]
      lines_tracker: dict(line_index -> { ... })
    """
    prompts_info = []
    lines_tracker = {}

    with open(input_file, "r", encoding="utf-8") as infile:
        for line_index, line in enumerate(tqdm(infile, desc="Reading data"), start=1):
            data = json.loads(line)
            
            conversations = data.get("conversations", [])
            num_rounds = len(conversations) // 2
            # 只要有一轮需要处理，就建个 entry
            lines_tracker[line_index] = {
                "data": data,
                "merged_conversations": [],
                "error_found": False,
                "total_rounds": num_rounds,       # 该行共有多少轮
                "completed_rounds": 0            # 已处理轮数
            }

            for round_index in range(num_rounds):
                i = round_index * 2
                if i + 1 >= len(conversations):
                    continue

                # 解析human_turn, gpt_turn
                human_turn = conversations[i]
                human_question = human_turn.get('value', '')
                gpt_turn = conversations[i+1]
                gpt_response = gpt_turn.get("value", "")

                conv = [human_turn, gpt_turn]

                # 尝试提取 <SUMMARY>、<CAPTION> 与 <CONCLUSION>
                summary_match = re.search(r'<SUMMARY>(.*?)</SUMMARY>', gpt_response, re.DOTALL)
                caption_match = re.search(r'<CAPTION>(.*?)</CAPTION>', gpt_response, re.DOTALL)
                # reasoning_match = re.search(r'<REASONING>(.*?)</REASONING>', gpt_response, re.DOTALL)
                conclusion_match = re.search(r'<CONCLUSION>(.*?)</CONCLUSION>', gpt_response, re.DOTALL)

                if summary_match and caption_match and conclusion_match:
                    summary_text = summary_match.group(1).strip()
                    caption_text = caption_match.group(1).strip()
                    # reasoning_text = reasoning_match.group(1).strip()
                    conclusion_text = conclusion_match.group(1).strip()

                    # 构造 prompt
                    description_text = [summary_text, caption_text]
                    # description_text = f"Summary: {summary_text}\nCaption: {caption_text}"
                    prompt_description = generate_des_prompt(conv)

                    # 收集
                    prompts_info.append({
                        "line_index": line_index,
                        "round_index": round_index,
                        "description_text": description_text,
                        "prompt_qwen": prompt_description,
                        "conclusion_text": conclusion_text
                    })
                else:
                    # 如果少了 SUMMARY/CAPTION/CONCLUSION，就不收集prompt
                    pass
    
    return prompts_info, lines_tracker


# ==================== 2) 并发处理: 在线对每行进行合并和保存 ==================== #
def single_call_api_and_process(info, lines_tracker, output_file, lock):
    """
    当某轮次出错时，仅将“该出错轮次”保存为负例，而之前已完成的轮次保存为正例；
    若直到所有轮次都没出错，则整个对话写为正例。
    
    具体逻辑：
    1) 如果该行之前已被判定出错/写出，则跳过。
    2) 正常调用API并判断正确性
       - 如果正确，把此轮对话加入 merged_conversations 里，若已到行末，则整行写正例。
       - 如果出错：
         a) 把之前已成功的轮次当作一个“部分成功”的正例写入（如果有成功轮次的话）。
         b) 当前出错轮次单独写为负例。
         c) 从 lines_tracker 移除该行，不再处理后续轮次。
    """
    
    line_index      = info["line_index"]
    round_index     = info["round_index"]
    conclusion_text = info["conclusion_text"]
    description_text = info["description_text"]
    
    # 如果该行已经被移除或标记出错，就不再处理
    if line_index not in lines_tracker:
        return line_index, round_index, None

    line_info = lines_tracker[line_index]
    data = line_info["data"]
    image_path = data["image"]
    image_full_path = f"/mnt/ali-sh-1/dataset/redone_juicefs/huangwx_ali_shanghai/dataset/llava-r1/llava_cot_images/{image_path}"
    image_full_path = encode_image(image_full_path)
    conversations = data.get("conversations", [])
    
    # 计算在 conversations 中的人问/机器答位置
    i = round_index * 2
    if i + 1 >= len(conversations):
        return line_index, round_index, None

    human_turn = conversations[i]
    gpt_turn   = conversations[i+1]

    # ---------------------------
    # （1）调用 API 拿到结果
    # ---------------------------
    prompt_qwen = info["prompt_qwen"]  
    summary, caption = description_text

    # print(f"Qwen prompt:{prompt_qwen}")
    qwen_des_response = chat(model_qwen, image_full_path, prompt_qwen)
    new_caption = qwen_des_response
    # print(f"New caption:{new_caption}")

    description_text = f"<DESCRIPTION>Summary:{summary}\nCaption:{new_caption}</DESCRIPTION>"

    # ---------------------------
    # （2）先替换 <SUMMARY>/<CAPTION> 
    # ---------------------------
    original_gpt_response = gpt_turn.get("value", "")
    new_gpt_response = re.sub(
        r'<SUMMARY>.*?</CAPTION>',
        lambda m: f"{description_text}",
        original_gpt_response,
        flags=re.DOTALL
    )

    # 构造修改后的 GPT 回复
    gpt_turn_modified = gpt_turn.copy()
    gpt_turn_modified['value'] = new_gpt_response

    # ---------------------------
    # （4）加锁后更新 merged_conversations 并写文件
    # ---------------------------
    with lock:
        # 如果 lines_tracker 在此期间被移除，也跳过
        if line_index not in lines_tracker:
            return line_index, round_index, None

        line_info = lines_tracker[line_index]
        merged_conversations = line_info["merged_conversations"]

        merged_conversations.append(human_turn)
        merged_conversations.append(gpt_turn_modified)
        line_info["completed_rounds"] += 1

        # 如果所有轮次处理完，且都正确 -> 整行写入正例
        if line_info["completed_rounds"] >= line_info["total_rounds"]:
            data["conversations"] = merged_conversations
            output_file.write(json.dumps(data, ensure_ascii=False) + "\n")
            output_file.flush()

            # 移除该行
            del lines_tracker[line_index]

    return line_index, round_index


def concurrency_call_api_online_save(prompts_info, lines_tracker, output_file, max_workers=128):
    """
    使用 ThreadPoolExecutor 并发处理，每轮的结果处理完成后就写文件
    """
    # 事先打开文件，锁
    output_file = open(output_file, "w", encoding="utf-8")
    lock = threading.Lock()  # 用于锁写文件
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for info in prompts_info:
            # 每个 prompt 都提交一个任务
            fut = executor.submit(single_call_api_and_process, info, lines_tracker, output_file, lock)
            futures.append(fut)

        # 这里只是等待所有任务完成（由于我们在线写文件，所以不需要后续再处理）
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
            try:
                # 取结果看是否有异常
                line_index, round_index  = fut.result()
            except Exception as e:
                print("Error in thread:", e)

    output_file.close()

# ==================== 主入口 ==================== #
def main():
    input_file = "/mnt/ali-sh-1/usr/zhaizijie/huangwx_ali_shanghai/code/llava_r1_process/llava_cot_data/data_split/llava_cot_split1.jsonl"
    output_file = "/mnt/ali-sh-1/usr/zhaizijie/huangwx_ali_shanghai/code/llava_r1_process/llava_cot_data/result_split/llava_cot_cleaned_split1.jsonl"# input_file = "/data/jbh/train_example.jsonl"
    # output_file = "train_output.jsonl"
    
    # 1) 先收集 prompts_info 和 lines_tracker
    prompts_info, lines_tracker = collect_prompts_and_build_tracker(input_file)

    # 2) 并发处理 & 在线保存
    concurrency_call_api_online_save(
        prompts_info, 
        lines_tracker, 
        output_file, 
        max_workers=128
    )

    print("All done.")

if __name__ == "__main__":
    main()
    
    
    
# {"id": "1ade7413-398a-48bb-8e9c-d648669ef388", "image": "sqa/train/8125/image.png", "conversations": [{"from": "human", "value": "What is the capital of Alaska?\nContext: N/A\nOptions: (A) Seattle (B) Juneau (C) Saint Paul (D) Anchorage"}, {"from": "gpt", "value": "<DESCRIPTION>Summary:To solve this problem, I will describe the image, focus on identifying the state of Alaska, and use logical reasoning to determine its capital from the given options.\nCaption:The image depicts a map of the United States with all states shaded in green. Alaska is prominently displayed in the lower left corner, separate from the contiguous states. The map includes all 50 states, with Alaska clearly distinguished from the mainland. The question asks for the capital of Alaska, and the map visually confirms Alaska's location, aiding in the identification of its capital city.</DESCRIPTION>\n\n<REASONING> The map includes Alaska, which is located in the northwest region of North America, detached from the contiguous United States. The question asks for the capital of Alaska. From the provided options, I need to identify the correct capital city of the state. Seattle and Saint Paul are known capitals of other locations, and Anchorage, while a major city in Alaska, is not the capital. Juneau is the correct capital of Alaska. </REASONING>\n\n<CONCLUSION> Juneau is the capital of Alaska. The answer is B. </CONCLUSION>"}]}
# {"id": "50b055f6-9ba2-4200-a65a-f70d80919fac", "image": "sqa/train/253/image.png", "conversations": [{"from": "human", "value": "What did the scientists discover?\nContext: Read the passage about a new discovery.\nA few scientists were looking for sharks when they saw something surprising. They found a sea turtle that glowed! The turtle's shell was bright red and green. This was a new discovery. Scientists had never seen a sea turtle with a glowing shell before.\nScientists want to know why these turtles have a shell that glows. Sadly, there are not many of these turtles left in the world. So, it is hard to learn about them.\nOptions: (A) a sea turtle that can fly (B) a sea turtle with a glowing shell"}, {"from": "gpt", "value": "<DESCRIPTION>Summary:To solve the problem, I will first describe the image focusing on relevant details, then analyze the provided passage to find what the scientists discovered. I will conclude by matching it with the standard correct answer.\nCaption:The image shows a sea turtle swimming underwater with a diver in the background. The turtle's shell exhibits a bright red and green glow, which is unusual and striking. The shell's pattern and colors are vivid, suggesting a unique and previously unobserved characteristic. The turtle's head and front flippers are visible, and the overall scene is set in a clear blue aquatic environment.</DESCRIPTION>\n\n<REASONING> The passage provided describes a scientific observation. The scientists, who were initially searching for sharks, unexpectedly discovered a sea turtle with a glowing shell, which was bright red and green. This discovery of a glowing shell is notable because it was previously unknown to scientists. This aligns with the standard answer, which specifies the discovery of a sea turtle with a glowing shell. </REASONING>\n\n<CONCLUSION> The answer is B. </CONCLUSION>"}]}
# {"id": "101d75e8-f0c2-4095-9caf-47786024e333", "image": "coco/train2017/000000136235.jpg", "conversations": [{"from": "human", "value": "What color is the FedEx airplane's tail fin?\nA. blue\nB. green\nC. yellow\nD. purple"}, {"from": "gpt", "value": "<DESCRIPTION>Summary:To solve the problem, I will first examine the image to identify the airplane associated with FedEx, focusing on its tail fin's color. Then, using this information, I will determine the correct answer by matching it with the given options.\nCaption:The image shows an airport runway with multiple airplanes. In the background, a FedEx plane is clearly visible. The tail fin of this FedEx airplane is a prominent feature to be noted. The FedEx plane's tail fin is purple, which is a distinctive color associated with the FedEx brand. This detail is crucial for identifying the correct answer to the question about the color of the FedEx airplane's tail fin.</DESCRIPTION>\n\n<REASONING> To determine the color of the FedEx plane's tail fin, I will identify the airplane with FedEx branding and observe the tail fin. Comparing the observed color with the options given (A. blue, B. green, C. yellow, D. purple) will allow me to choose the correct answer. </REASONING>\n\n<CONCLUSION> D </CONCLUSION>"}, {"from": "human", "value": "What does the sign in front of the runways near the camera say?\nA. one way\nB. stop\nC. dead end\nD. enter\nAnswer with the option's letter from the given choices directly."}, {"from": "gpt", "value": "<DESCRIPTION>Summary:To solve the problem, I will carefully analyze the image to identify the sign near the camera. I will focus on reading the text of the sign as it appears in the image and select the option that matches exactly.\nCaption:The image depicts an airport runway with a plane landing in the background. In the foreground, there is a prominent \"Do Not Enter\" sign with white text on a red background. The sign is clearly visible and positioned near the runways. The airport terminal and other planes are visible in the distance, along with some greenery and clear skies. The sign's text is legible and matches the options provided in the question.</DESCRIPTION>\n\n<REASONING> First, identify the sign in the foreground of the image and read the text on it. The text states \"Do Not Enter,\" which suggests restrictions on access to the runway from this direction. Review the given answer options and determine which matches the information on the sign. </REASONING>\n\n<CONCLUSION> D </CONCLUSION>"}]}