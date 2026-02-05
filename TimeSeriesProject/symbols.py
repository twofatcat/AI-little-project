"""
选股接口：get_symbols(market, theme) -> list[str]
当前硬编码中美 AI/机器人 全产业链各 20 只；后期可改为调用 LLM API 返回列表。
"""
from typing import List

# US: AI + 机器人 硬核产业链（算力芯片/人形机器人/工业自动化/软件）
# 剔除了纯消费电子，聚焦 B 端与基础设施
SYMBOLS_US_AI_ROBOTICS: List[str] = [
    # --- The Brain: AI Chips & Compute (算力/芯片) ---
    "NVDA",  # NVIDIA: AI 算力绝对霸主
    "AMD",   # AMD: GPU 第二把交椅
    "INTC",  # Intel: 正在发力代工与AI芯片
    "TSM",   # TSMC (台积电): 所有 AI 芯片的制造者 (ADR)
    "AVGO",  # Broadcom: 定制 AI 芯片与网络传输
    "MU",    # Micron: HBM 高带宽内存 (AI 必不可少)

    # --- The Body: Robotics & Automation (机器人本体/自动化) ---
    "TSLA",  # Tesla: Optimus 人形机器人 + FSD 自动驾驶
    "ROK",   # Rockwell Automation: 美国工业自动化龙头
    "TER",   # Teradyne: 拥有 Universal Robots (协作机器人鼻祖)
    "ISRG",  # Intuitive Surgical: 达芬奇手术机器人 (医疗机器人龙头)
    "SYM",   # Symbotic: 仓储物流机器人 (沃尔玛投资)
    "DE",    # John Deere: 农业自动驾驶机器人巨头

    # --- The Soul: AI Software & Cloud (大模型/云/数据) ---
    "MSFT",  # Microsoft: OpenAI 的金主，Copilot
    "GOOGL", # Alphabet: DeepMind, Waymo, Gemini
    "META",  # Meta: PyTorch 框架, 开源 Llama 模型
    "PLTR",  # Palantir: AI 大数据操作系统
    "PATH",  # UiPath: RPA 软件机器人 (流程自动化)
    
    # --- Upstream Tools (上游设备/软件) ---
    "AMAT",  # Applied Materials: 芯片制造设备
    "CDNS",  # Cadence: 芯片设计软件 (EDA)
    "SNPS",  # Synopsys: 芯片设计软件 (EDA)
]

# CN: 沪深京 A股，聚焦 减速器/电机/传感器/AI芯片/服务器
SYMBOLS_CN_AI_ROBOTICS: List[str] = [
    # --- 核心零部件：关节与运动控制 (Actuators & Motion Control) ---
    "300124", # 汇川技术: 工控之王，伺服电机与驱动国内第一
    "688017", # 绿的谐波: 谐波减速器 (机器人关节核心)，打破日本垄断
    "002472", # 双环传动: RV减速器 (重载机器人关节)
    "603728", # 鸣志电器: 空心杯电机 (人形机器人灵巧手专用)
    "002050", # 三花智控: 特斯拉人形机器人执行器主要供应商
    "601689", # 拓普集团: 机器人直线执行器与底盘

    # --- 大脑：AI 芯片与算力 (AI Chips & Infra) ---
    "688256", # 寒武纪: 国产 AI 训练芯片龙头
    "688041", # 海光信息: 国产 X86 CPU + DCU (类CUDA生态)
    "603019", # 中科曙光: AI 服务器与液冷数据中心
    "002371", # 北方华创: 芯片制造设备龙头 (上游铲子股)
    
    # --- 感知：眼睛与传感器 (Vision & Sensors) ---
    "002415", # 海康威视: 机器视觉与 AI 感知
    "688322", # 奥比中光: 3D 视觉传感器 (机器人的眼睛)
    "002920", # 德赛西威: 智能驾驶域控制器 (边缘 AI)

    # --- 本体与系统集成 (Robot Body & Integrators) ---
    "002747", # 埃斯顿: 国产工业机器人本体出货量第一
    "300024", # 机器人 (新松): 中科院背景，老牌机器人巨头
    "688169", # 石头科技: 扫地机器人 (消费级自动驾驶落地)

    # --- 软件与应用 (Software & Application) ---
    "002230", # 科大讯飞: 语音交互与认知大模型
    "300496", # 中科创达: 智能终端 OS，机器人操作系统
    "601360", # 三六零: AI 搜索与安全 (你特别提到的)
    "688111", # 金山办公: AI 办公应用落地 (WPS AI)
]


def get_symbols(market: str, theme: str = "ai_robotics") -> List[str]:
    """
    Return symbol list for the given market and theme.
    Open upstream interface: later replace with LLM API call.
    """
    if market == "us":
        if theme == "ai_robotics":
            return list(SYMBOLS_US_AI_ROBOTICS)
        return list(SYMBOLS_US_AI_ROBOTICS)
    if market == "cn":
        if theme == "ai_robotics":
            return list(SYMBOLS_CN_AI_ROBOTICS)
        return list(SYMBOLS_CN_AI_ROBOTICS)
    return []
