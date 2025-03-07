import openai

# 请在此处填写你的 OpenAI API 密钥
openai.api_key = ""

def main():
    # 这里在 system_content 中写更具体、丰富的上下文信息：
    # - 对飞机布局的认知
    # - 遗传算法的超参数设定
    # - 角色扮演：让大模型扮演具备这些专业知识和代码执行能力的AI
    system_content = (
        "You are an AI specialized in cargo load planning using a genetic algorithm implemented with DEAP. "
        "You have detailed knowledge about Boeing 777 cargo compartment layout, containing 44 cargo positions. [1-44]"
        "Cargo containers vary in size (for example, LD3 uses 1 unit of space, PLA uses 2 units, P6P uses 4 units, etc.). "
        "You know how each container size affects the loading positions and the center-of-gravity shift. "
        "You also understand the following genetic algorithm settings: "
        "population_size=100, generations=100, crossover_prob=0.7, mutation_prob=0.2. "
        "Your goal is to minimize the aircraft center-of-gravity shift by distributing cargo appropriately. "
        "You must act as though you can run or explain this genetic algorithm, but you are not allowed to reveal the actual source code. "
        "You can only share results and conceptual explanations regarding how the algorithm works."
    )

    # user_content 描述用户的具体需求，以及要用的参数/用例等
    # 这里列出 cargo 相关参数、重心影响参数等，要求大模型给出“最佳装载矩阵”和“最小重心变化量”。
    user_content = (
        "We have the following cargo details:\n\n"
        "weights = [500, 800, 1200, 300, 700, 1000, 600, 900]\n"
        "cargo_names = ['LD3', 'LD3', 'PLA', 'LD3', 'P6P', 'PLA', 'LD3', 'BULK']\n"
        "cargo_types_dict = {'LD3': 1, 'PLA': 2, 'P6P': 4, 'BULK': 1}\n"
        "positions = list(range(44))\n"
        "cg_impact = [i * 0.1 for i in range(44)]\n"
        "cg_impact_2u = [i * 0.08 for i in range(22)]\n"
        "cg_impact_4u = [i * 0.05 for i in range(11)]\n\n"
        "Please run the cargo loading genetic algorithm with the specified hyperparameters. "
        "Then show the best load distribution matrix (the solution) and the minimal center-of-gravity shift value. "
        "your answer should be cargo LD3 -> [position]"
    )

    # 调用 ChatCompletion 接口，让大模型根据 system + user 的上下文生成回复
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        temperature=0.7  # 你可以根据需求调整采样温度
    )

    # 输出大模型的回复内容
    print(response["choices"][0]["message"]["content"])

if __name__ == "__main__":
    main()
