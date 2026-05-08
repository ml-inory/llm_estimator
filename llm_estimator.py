import json
import argparse


def estimate_hardware(model_params, target_latency_ms):
    P = model_params['P']  # e.g., 7e9
    Q = model_params['Q']  # e.g., 0.5 (4bit)
    H = model_params['H']  # e.g., 4096
    L = model_params['L']  # e.g., 2048
    B = model_params['B']  # e.g., 1
    N_layer = model_params['N_layer']   # e.g., 32
    
    # 1. 计算Decode阶段所需带宽 (GB/s)
    kv_cache_size = 2 * L * H * N_layer * 2 / 1e9  # FP16 KV cache in GB
    weight_size = P * Q / 1e9  # Weight in GB
    total_mem_per_token = weight_size + kv_cache_size
    bandwidth_req = total_mem_per_token / (target_latency_ms / 1000)
    
    # 2. 计算所需算力 (TOPS)
    ops_per_token = 2 * P * Q  # MAC ops
    tops_req = ops_per_token / 1e12 / (target_latency_ms / 1000)
    
    # 3. 最小SRAM建议 (MB)
    sram_min_mb = (kv_cache_size * 1024) * 0.2  # 存20% KV cache活跃部分

    return bandwidth_req, tops_req, sram_min_mb


def main():
    parser = argparse.ArgumentParser(description="根据模型配置文件估算硬件需求")
    parser.add_argument("config_file", type=str, help="模型配置JSON文件路径")
    parser.add_argument("--tpot", "-t", type=float, default=50, 
                        help="目标TPOT延迟 (ms/token), 默认50ms")

    args = parser.parse_args()

    # 读取配置文件
    with open(args.config_file, 'r') as f:
        model_params = json.load(f)

    # 确保P是原始数值（不是Billion）
    # 如果配置文件中P是Billion单位（如7表示7B），需要转换
    if 'P' in model_params and model_params['P'] < 1000:
        # 假设小于1000的P值是以Billion为单位
        model_params['P'] = model_params['P'] * 1e9
    
    # 确保Q正确计算（如果是整数位宽，转换为字节）
    if 'Q' in model_params and model_params['Q'] > 2:
        # 假设Q是位宽（如4, 8, 16），转换为字节
        bits = model_params['Q']
        model_params['Q'] = bits / 8

    bandwidth_req, tops_req, sram_min_mb = estimate_hardware(model_params, args.tpot)
    print(f"所需带宽: {bandwidth_req:.2f} GB/s")
    print(f"所需算力: {tops_req:.2f} TOPS")
    print(f"建议SRAM: {sram_min_mb:.2f} MB")


if __name__ == "__main__":
    main()