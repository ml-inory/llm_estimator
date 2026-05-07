import json
import argparse
from transformers import AutoConfig


def get_model_config(model_name: str, seq_length: int = 2048, batch_size: int = 1, quant_bits: int = 4):
    """
    从HuggingFace加载模型配置，输出硬件估算所需的参数
    
    Args:
        model_name: HuggingFace模型名称，如 "meta-llama/Llama-2-7b-hf"
        seq_length: 目标序列长度 (L)
        batch_size: 批处理大小 (B)
        quant_bits: 量化位数，如 4 表示 INT4, 8 表示 INT8, 16 表示 FP16
    
    Returns:
        dict: 包含模型负载特征的字典
    """
    # 加载配置
    config = AutoConfig.from_pretrained(model_name)
    
    # 提取参数
    hidden_size = getattr(config, 'hidden_size', None)
    num_attention_heads = getattr(config, 'num_attention_heads', None)
    num_key_value_heads = getattr(config, 'num_key_value_heads', num_attention_heads)  # MHA默认等于num_heads
    num_hidden_layers = getattr(config, 'num_hidden_layers', None)
    
    # 估算参数量 (Billion)
    # 方法1: 从config中直接读取（某些模型提供）
    if hasattr(config, 'total_params'):
        total_params_b = config.total_params / 1e9
    else:
        # 方法2: 估算
        # Embedding + LM Head 参数
        vocab_size = getattr(config, 'vocab_size', 32000)
        tie_word_embeddings = getattr(config, 'tie_word_embeddings', False)
        
        embedding_params = vocab_size * hidden_size
        lm_head_params = 0 if tie_word_embeddings else vocab_size * hidden_size
        
        # 每层参数 (简化估算)
        # Attention: Q,K,V,O 四个投影矩阵
        attn_params = 4 * hidden_size * hidden_size
        # FFN: 通常两个线性层，intermediate_size 可能是 4*hidden_size 或其他值
        intermediate_size = getattr(config, 'intermediate_size', 4 * hidden_size)
        ffn_params = 2 * hidden_size * intermediate_size
        # LayerNorm/RMSNorm 参数（通常可忽略）
        norm_params = 2 * hidden_size  # 每层两个norm，但很多实现不包含可学习参数
        
        per_layer_params = attn_params + ffn_params + norm_params
        
        total_params = embedding_params + lm_head_params + num_hidden_layers * per_layer_params
        total_params_b = total_params / 1e9
    
    # 参数名称处理
    model_name_clean = model_name.split('/')[-1]
    
    # 构建输出
    result = {
        "name": model_name_clean,
        "P": round(total_params_b, 2),           # 参数量 (Billion)
        "Q": quant_bits / 8,                      # 每参数字节数 (INT4=0.5, INT8=1, FP16=2)
        "H": hidden_size,
        "L": seq_length,
        "B": batch_size,
        "N_h": num_attention_heads,
        "N_layer": num_hidden_layers,
        # 附加信息
        "kv_heads": num_key_value_heads,          # GQA场景下KV头数
        "intermediate_size": intermediate_size if 'intermediate_size' in dir() else None,
        "vocab_size": vocab_size,
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="从HuggingFace提取模型负载特征")
    parser.add_argument("model_name", type=str, help="HuggingFace模型名称")
    parser.add_argument("--seq-length", "-l", type=int, default=2048, help="序列长度 (默认2048)")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="批处理大小 (默认1)")
    parser.add_argument("--quant-bits", "-q", type=int, default=4, choices=[4, 8, 16], 
                        help="量化位数: 4=INT4, 8=INT8, 16=FP16 (默认4)")
    parser.add_argument("--output", "-o", type=str, help="输出JSON文件路径 (不指定则打印到控制台)")
    
    args = parser.parse_args()
    
    # 获取配置
    result = get_model_config(
        args.model_name,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        quant_bits=args.quant_bits
    )
    
    # 输出
    json_str = json.dumps(result, indent=4, ensure_ascii=False)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(json_str)
        print(f"配置文件已保存到: {args.output}")
    else:
        print(json_str)


if __name__ == "__main__":
    main()