from datasets import load_dataset

SYSTEM_PROMPT = ""
PREFIX_USER_PROMPT = """
Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output."""


if __name__ == "__main__":
    dataset = load_dataset("open-thoughts/OpenThoughts-114k", 
                           name="metadata", 
                           split="train")
    print(dataset)
    # 打印所有的键
    print(dataset.column_names)
    
    # 选择 domain 为 "code" 的数据
    code_domain_data = dataset.filter(lambda x: x["domain"] == "code")
    # 选择 deepseek_reasoning 与 deepseek_solution 长度之和 < 64000 个字符的数据
    code_domain_data = code_domain_data.filter(lambda x: len(x["deepseek_reasoning"] + x["deepseek_solution"]) < 64000)     # 剩余 17946
    print(code_domain_data)
    
    # 对 deepseek_solution中的代码，提取出代码块，并使用test_cases中的输入输出进行测试，保留测试通过的样本
    def extract_code_and_test(sample):
        pass