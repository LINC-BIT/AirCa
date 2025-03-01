import openai

# 请在此处填写你的 OpenAI API 密钥
openai.api_key = "YOUR_OPENAI_API_KEY"

def main():
    # ---- 系统角色: 告诉大模型它拥有PSO算法，并设定它的身份和知识范围 ----
    system_content = (
        "You are an AI specialized in cargo load planning using a PSO algorithm implemented with PySwarms. "
        "You have detailed knowledge about Boeing 777 cargo compartment layout, which has 44 cargo positions. "
        "Containers come in different sizes, for example LD3 (1 unit), PLA (2 units), P6P (4 units), etc. "
        "You also understand the hyperparameters for PSO: swarmsize=100, maxiter=100, as well as c1=0.5, c2=0.3, w=0.9. "
        "Your goal is to minimize the aircraft center-of-gravity (CG) shift by distributing cargo properly. "
        "You must act as though you can run or explain the code, but you can't reveal the actual source code. "
        "Please focus on providing the results or conceptual explanations."
    )

    # ---- 用户角色: 提供具体的输入场景和需求 ----
    user_content = (
        "We want to run a binary PSO-based cargo load planning with the following inputs:\n"
        "weights = [500, 800, 1200, 300, 700, 1000, 600, 900]\n"
        "cargo_names = ['LD3', 'LD3', 'PLA', 'LD3', 'P6P', 'PLA', 'LD3', 'BULK']\n"
        "cargo_types_dict = {'LD3': 1, 'PLA': 2, 'P6P': 4, 'BULK': 1}\n"
        "positions = list(range(44))\n"
        "cg_impact = [i * 0.1 for i in range(44)]\n"
        "cg_impact_2u = [i * 0.08 for i in range(22)]\n"
        "cg_impact_4u = [i * 0.05 for i in range(11)]\n"
        "max_positions = 44\n\n"
        "Please run the PSO algorithm with swarmsize=100, maxiter=100, and the options c1=0.5, c2=0.3, w=0.9. "
        "Provide the best cargo loading solution matrix and the minimal center-of-gravity shift. "
        "If you have any advice on improving the result, please share it as well."
    )

    # ---- 调用OpenAI ChatCompletion接口 ----
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        temperature=0.7
    )

    # ---- 打印大模型回复 ----
    print(response["choices"][0]["message"]["content"])

if __name__ == "__main__":
    main()
