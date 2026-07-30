import base64
import os
from PIL import Image, ImageOps
import io
from tqdm import tqdm

# ========== 模型配置 ==========
# ========== 清除所有代理环境变量（防止 httpx 报 socks:// 错误）==========
for proxy_var in [
    'http_proxy', 'https_proxy', 'all_proxy', 'ALL_PROXY',
    'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY', 'no_proxy'
]:
    os.environ.pop(proxy_var, None)
# =======================================================================
TEMP=0.5
TOP_P=0.85
SUPPORTED_AGENTS = {
    # === 多模态模型（支持图像）===
    "local-gemma3-27b": {"type": "ollama", "model": "gemma3:27b", "multimodal": True},
    "local-qwen2.5vl-7b": {"type": "ollama", "model": "qwen2.5vl:7b", "multimodal": True},
    "local-qwen2.5vl-32b": {"type": "ollama", "model": "qwen2.5vl:32b", "multimodal": True},
    "local-qwen3-32b": {"type": "ollama", "model": "qwen3:32b", "multimodal": False},
    "local-llama3.1:8b": {"type": "ollama", "model": "llama3.1:8b", "multimodal": False},
    "local-deepseek-r1:14b": {"type": "ollama", "model": "deepseek-r1:14b", "multimodal": False},
    "lightai-gpt-4o": {
        "type": "openai", "model": "gpt-4o",
        "api_key_env": "LIGHTAI_API_KEY",
        "base_url": "https://api.lightai.io/v1",
        "multimodal": True
    },
    "lightai-gpt-5": {
        "type": "openai", "model": "gpt-5-chat-latest",
        "api_key_env": "LIGHTAI_API_KEY",
        "base_url": "https://api.lightai.io/v1",
        "multimodal": True
    },
    "glm-4v": {
        "type": "zhipu", "model": "glm-4v",
        "api_key_env": "ZHIPUAI_API_KEY",
        "multimodal": True
    },
    "gpt-4o-mini": {
        "type": "openai", "model": "gpt-4o-mini",
        "api_key_env": "BLTCY_API_KEY",
        "base_url": "https://api.bltcy.ai/v1",
        "multimodal": True
    },

    # === 纯文本模型 ===
    "glm-4": {
        "type": "zhipu", "model": "glm-4",
        "api_key_env": "ZHIPUAI_API_KEY",
        "multimodal": False
    },

    "local-llama3": {
        "type": "ollama", "model": "llama3:latest",
        "multimodal": False
    },
}

# ========== 统一客户端管理 ==========

clients = {}  # 缓存已初始化的客户端


def get_client(agent: str):
    """统一懒加载客户端，按需初始化，安全导入"""
    global clients
    if agent in clients:
        return clients[agent]

    cfg = SUPPORTED_AGENTS[agent]
    client_type = cfg["type"]
    api_key = os.getenv(cfg.get("api_key_env", "")) if cfg.get("api_key_env") else None
    try:
        if client_type == "ollama":
            from ollama import Client
            client = Client(host="http://127.0.0.1:11434")
            clients[agent] = client
            return client

        elif client_type == "openai":
            from openai import OpenAI
            if not api_key:
                raise RuntimeError(f"Set {cfg['api_key_env']} before using {agent}")
            client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
            clients[agent] = client
            return client

        elif client_type == "zhipu":
            from zhipuai import ZhipuAI
            if not api_key:
                raise RuntimeError(f"Set {cfg['api_key_env']} before using {agent}")
            client = ZhipuAI(api_key=api_key)
            clients[agent] = client
            return client

        else:
            raise ValueError(f"Unsupported client type: {client_type}")

    except ImportError as e:
        pkg = {
            "ollama": "ollama",
            "openai": "openai",
            "zhipu": "zhipuai"
        }.get(client_type, client_type)
        raise ImportError(f"请安装依赖包: pip install {pkg}") from e
    except Exception as e:
        raise RuntimeError(f"初始化 {agent} 客户端失败: {str(e)}") from e


# ========== 工具函数 ==========

def encode_image(image_path: str, resize: bool = False, target_size: int = 512) -> str:
    """将图像编码为 Base64 字符串（仅用于 VLM）"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        if resize:
            w, h = img.size
            if w > h:
                new_w, new_h = target_size, int(h * target_size / w)
            else:
                new_h, new_w = target_size, int(w * target_size / h)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ========== 核心接口 ==========

def llm_query(system_prompt: str, user_query: str, agent: str = "glm-4", timeout: int = 30) -> str:
    """纯文本 LLM 调用接口"""
    if agent not in SUPPORTED_AGENTS:
        raise ValueError(f"Unsupported agent: {agent}")

    cfg = SUPPORTED_AGENTS[agent]
    client = get_client(agent)

    try:
        if cfg["type"] == "ollama":
            response = client.chat(
                model=cfg["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                options={"temperature": TEMP, "top_p": TOP_P}
            )
            answer = response["message"]["content"]

        elif cfg["type"] == "openai":
            response = client.chat.completions.create(
                model=cfg["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                timeout=timeout
            )
            answer = response.choices[0].message.content

        elif cfg["type"] == "zhipu":
            response = client.chat.completions.create(
                model=cfg["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                timeout=timeout
            )
            answer = response.choices[0].message.content

        else:
            raise NotImplementedError(f"Agent type '{cfg['type']}' not implemented for text.")

        tqdm.write(f"[ANSWER]: {answer}")
        return answer

    except Exception as e:
        tqdm.write(f"[ERROR] {str(e)}")
        raise


def vlm_query(system_prompt: str, image_path: str, user_query: str, agent: str = "local-qwen2.5vl-7b", resize: bool = True, target_size: int = 512) -> str:
    """视觉语言模型调用接口（图文）"""
    if agent not in SUPPORTED_AGENTS:
        raise ValueError(f"Unsupported agent: {agent}")
    if not SUPPORTED_AGENTS[agent].get("multimodal", False):
        raise ValueError(f"Agent '{agent}' 不支持多模态输入，请使用纯文本接口 llm_query()")

    cfg = SUPPORTED_AGENTS[agent]
    base64_image = encode_image(image_path, resize=resize, target_size=target_size)
    tqdm.write(f"[SYSTEM]: {system_prompt}")
    tqdm.write(f"[QUERY]: {user_query}")

    try:
        client = get_client(agent)

        if cfg["type"] == "ollama":
            response = client.chat(
                model=cfg["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query, "images": [base64_image]}
                ],
                options={"temperature": TEMP, "top_p": TOP_P}
            )
            answer = response["message"]["content"]

        elif cfg["type"] == "openai":
            image_url = f"data:image/jpeg;base64,{base64_image}"
            response = client.chat.completions.create(
                model=cfg["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_query},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                timeout=25
            )
            answer = response.choices[0].message.content

        elif cfg["type"] == "zhipu":
            # GLM-4V 要求图像和文本分两条消息
            response = client.chat.completions.create(
                model=cfg["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"image/jpeg;base64,{base64_image}"}
                            }
                        ]
                    },
                    {"role": "user", "content": user_query}
                ],
                timeout=60
            )
            answer = response.choices[0].message.content

        else:
            raise NotImplementedError(f"Agent type '{cfg['type']}' not implemented for VLM.")

        tqdm.write(f"[ANSWER]: {answer}")
        return answer

    except Exception as e:
        tqdm.write(f"[ERROR] {str(e)}")
        raise


# ========== 测试入口 ==========

if __name__ == "__main__":
    # === 测试纯文本 ===
    print("\n" + "="*50)
    print("🔍 测试纯文本 LLM：gpt-4o-mini")
    text_result = llm_query(
        system_prompt="You are a rigorous assistant, please answer in one sentence.",
        user_query="Is the earth round?",
        agent="gpt-4o-mini"
    )
    print("✅ 纯文本回答:", text_result)

    # === 测试多模态（如配置了图像路径）===
    image_path = os.path.join("pipeline", "sg", "fig", "allensville_scene_graph_3d.png")
    if os.path.exists(image_path):
        print("\n" + "="*50)
        print("🖼️  测试多模态 VLM：gpt-4o-mini")
        vlm_result = vlm_query(
            system_prompt="You are a visual assistant. Please describe the image content.",
            image_path=image_path,
            user_query="What objects are in the picture?",
            agent="gpt-4o-mini"
        )
        print("✅ VLM 回答:", vlm_result)
    else:
        print(f"⚠️  图像路径不存在，跳过多模态测试: {image_path}")
