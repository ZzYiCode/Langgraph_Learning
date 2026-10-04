


def load_prompt(file_path:str):
    with open(file_path, "r", encoding="utf-8") as f:
        prompt = f.read()
    return prompt


construction_prompt = load_prompt("./prompts/construction_prompt.txt")
write_prompt = load_prompt("./prompts/write_prompt.txt")
if __name__ == "__main__":
    print(construction_prompt)
